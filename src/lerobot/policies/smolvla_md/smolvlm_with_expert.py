# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput


class TemporalSmolVLMEncoder(nn.Module):
    """
    Drop-in replacement for SmolVLMEncoder that adds MEM-style space-time separable
    temporal attention (Option A+).

    Architecture (MEM paper Appendix C):
      - All 12 ViT layers run standard bidirectional spatial attention unchanged.
      - At layers in `temporal_layers` (every 4th by default: 3, 7, 11), we
        ADDITIONALLY run causal temporal attention across K frames reusing the
        existing W_q / W_k / W_v / W_o of that layer — no new weight matrices.
      - The temporal contribution is gated by a per-layer learnable scalar
        `temporal_alpha`, initialised to 0 (Option A+):
            h = spatial_residual + temporal_alpha * temporal_out
        At init alpha=0 → identical to vanilla SmolVLA for any K.
      - Fixed sinusoidal temporal position encoding e(t) with e(0)=0 for the
        current frame is added to hidden states before each temporal layer's LN.

    Usage:
        enc = TemporalSmolVLMEncoder(vision_model.encoder, temporal_layers=[3,7,11])
        vision_model.encoder = enc          # inject into frozen vision model
        enc.n_frames = K                    # set before forward; default 1
        enc.frame_indices = [-(K-1),..,0]  # set before forward; default [0]
    """

    def __init__(self, orig_encoder, temporal_layers=(3, 7, 11)):
        super().__init__()
        # Reuse the existing (frozen) ViT layers — weights NOT copied, only referenced.
        self.layers = orig_encoder.layers
        self.temporal_layers = set(temporal_layers)
        self._sorted_temporal = sorted(temporal_layers)
        self._layer_to_alpha = {l: i for i, l in enumerate(self._sorted_temporal)}

        # One learnable scalar per temporal layer, init=0 → no effect at start.
        # These are the ONLY new parameters added by SmolVLA-M (3 scalars total).
        self.temporal_alpha = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in self._sorted_temporal]
        )

        # Runtime context set externally before each forward call.
        self.n_frames: int = 1
        self.frame_indices: list[int] = [0]

    # ------------------------------------------------------------------
    # Fixed sinusoidal temporal PE  (e(0) = 0 by construction)
    # ------------------------------------------------------------------

    def _temporal_pe(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Returns (K, D) sinusoidal PE; row for t=0 is exactly zero."""
        K = self.n_frames
        indices = self.frame_indices          # list of K ints, last one is 0
        D = self.layers[0].embed_dim

        t = torch.tensor(indices, dtype=torch.float32, device=device)   # (K,)
        freqs = torch.pow(
            10000.0,
            -torch.arange(0, D, 2, dtype=torch.float32, device=device) / D
        )                                                                  # (D/2,)
        args = t[:, None] * freqs[None, :]                               # (K, D/2)
        pe = torch.zeros(K, D, dtype=torch.float32, device=device)
        pe[:, 0::2] = torch.sin(args)
        pe[:, 1::2] = torch.cos(args)
        pe[-1] = 0.0   # explicit zero for current frame (t=0), overrides cos(0)=1
        return pe.to(dtype)

    # ------------------------------------------------------------------
    # Causal temporal attention (reuses frozen spatial W_q/k/v/o)
    # ------------------------------------------------------------------

    def _temporal_attn(
        self,
        layer,
        h_norm: torch.Tensor,   # (B*K, N, D)  already LN'd
        B: int,
        K: int,
        N: int,
    ) -> torch.Tensor:
        """
        For every patch position p, attends over K timesteps with a causal mask.
        Reuses layer.self_attn projections — no new weights.
        Returns (B*K, N, D).
        """
        sa = layer.self_attn
        n_heads, head_dim, D = sa.num_heads, sa.head_dim, sa.embed_dim

        # Project with frozen spatial weights
        q = sa.q_proj(h_norm)   # (B*K, N, D)
        k = sa.k_proj(h_norm)
        v = sa.v_proj(h_norm)

        # Reshape: spatial (B*K, N, D) → temporal (B*N, K, n_heads, head_dim)
        def to_temporal(x):
            # (B*K, N, D) → (B, K, N, D) → (B, N, K, D) → (B*N, K, n_heads, head_dim)
            x = x.view(B, K, N, D).permute(0, 2, 1, 3)          # (B, N, K, D)
            x = x.reshape(B * N, K, n_heads, head_dim)
            return x.permute(0, 2, 1, 3).contiguous()            # (B*N, n_heads, K, head_dim)

        q, k, v = to_temporal(q), to_temporal(k), to_temporal(v)

        # Causal mask: lower-triangular — current frame (last) attends to all past
        causal = torch.tril(torch.ones(K, K, device=h_norm.device, dtype=torch.bool))

        scale = math.sqrt(head_dim) ** -1
        scores = (q @ k.transpose(-2, -1)) * scale               # (B*N, n_heads, K, K)
        scores = scores.masked_fill(~causal[None, None], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                            # (B*N, n_heads, K, head_dim)

        # Back to spatial layout: (B*N, n_heads, K, head_dim) → (B*K, N, D)
        out = out.permute(0, 2, 1, 3).contiguous()               # (B*N, K, n_heads, head_dim)
        out = out.view(B * N, K, D)                              # (B*N, K, D)
        out = out.permute(1, 0, 2).contiguous()                  # (K, B*N, D)
        out = out.view(K, B, N, D).permute(1, 0, 2, 3)          # (B, K, N, D)
        out = out.contiguous().view(B * K, N, D)                 # (B*K, N, D)

        return sa.out_proj(out)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        hidden_states = inputs_embeds
        K = self.n_frames
        BK, N, D = hidden_states.shape
        B = BK // K

        # Temporal PE: (K, D) → broadcast to (B*K, N, D)
        if K > 1:
            pe = self._temporal_pe(hidden_states.device, hidden_states.dtype)  # (K, D)
            pe = pe[:, None, :].expand(K, N, D)                                # (K, N, D)
            tpe = pe.unsqueeze(0).expand(B, K, N, D).reshape(BK, N, D)        # (B*K, N, D)
        else:
            tpe = None

        for idx, layer in enumerate(self.layers):
            if idx in self.temporal_layers and K > 1:
                alpha = self.temporal_alpha[self._layer_to_alpha[idx]]
                orig_dtype = hidden_states.dtype

                # --- Spatial attention (manual to intercept before MLP) ---
                residual = hidden_states
                h_with_tpe = hidden_states + tpe                    # add temporal PE
                h_norm = layer.layer_norm1(h_with_tpe)
                spatial_out, _ = layer.self_attn(
                    hidden_states=h_norm,
                    attention_mask=attention_mask,
                )
                # --- Temporal attention (same W_q/k/v, causal, gated by alpha) ---
                temporal_out = self._temporal_attn(layer, h_norm, B, K, N)
                # Cast alpha (Float32 param) and temporal_out to hidden_states dtype to
                # prevent Float32/BFloat16 type promotion from corrupting the residual stream.
                hidden_states = residual + spatial_out + alpha.to(orig_dtype) * temporal_out.to(orig_dtype)

                # --- MLP (unchanged) ---
                residual = hidden_states
                hidden_states = layer.layer_norm2(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states
            else:
                hidden_states = layer(hidden_states, attention_mask)

        return BaseModelOutput(last_hidden_state=hidden_states)


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
        temporal_vit_layers: tuple[int, ...] = (3, 7, 11),
        device: str = "auto",
    ):
        super().__init__()
        if load_vlm_weights:
            print(f"Loading  {model_id} weights ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        self.processor = AutoProcessor.from_pretrained(model_id)
        if num_vlm_layers > 0:
            print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        # Smaller lm expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0, (
                f"Number of layers in the VLM {len(self.get_vlm_model().text_model.layers)} are not multiple of num_expert_layers {num_expert_layers}"
            )
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)

        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers
        if "cross" in attention_mode:
            # Reshape qkv projections to have the same input dimension as the vlm
            for layer_idx in range(len(self.lm_expert.layers)):
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size

        # Inject MEM-style temporal encoder into the ViT.
        # TemporalSmolVLMEncoder wraps the existing (frozen) ViT layers and adds
        # a causal temporal attention pass at `temporal_vit_layers`, gated by
        # learnable scalars (temporal_alpha) initialised to 0.
        orig_enc = self.get_vlm_model().vision_model.encoder
        self.temporal_enc = TemporalSmolVLMEncoder(
            orig_encoder=orig_enc,
            temporal_layers=temporal_vit_layers,
        )
        self.get_vlm_model().vision_model.encoder = self.temporal_enc

        self.set_requires_grad()

    def get_vlm_model(self):
        return self.vlm.model

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        # temporal_alpha must always be trainable — re-enable AFTER all freeze passes
        # because train_expert_only would otherwise freeze them together with self.vlm.
        for param in self.temporal_enc.temporal_alpha.parameters():
            param.requires_grad = True
        # To avoid unused params issue with distributed training
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()

    def embed_image(self, image: torch.Tensor, n_frames: int = 1):
        """
        Args:
            image:    (B*K, C, H, W)  — K frames per batch item stacked in batch dim.
            n_frames: K, number of temporal frames (1 = vanilla single-frame behaviour).
        Returns:
            (B, num_tokens, D_lm) — tokens for the CURRENT frame only, after connector.
        """
        # Tell the temporal encoder how many frames are in this batch.
        enc = self.temporal_enc
        enc.n_frames = n_frames
        enc.frame_indices = list(range(-(n_frames - 1), 1))   # [-(K-1), ..., 0]

        # Forward through the full ViT (our TemporalSmolVLMEncoder is already injected).
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
            )
            .last_hidden_state
        )   # (B*K, N_patches, D_vit)

        # Drop past-frame tokens — keep only the current frame (index K-1 in each group).
        if n_frames > 1:
            image_hidden_states = image_hidden_states[n_frames - 1 :: n_frames]   # (B, N, D)

        # Modality projection & pixel-shuffle resampling: (B, N, D_vit) → (B, 64, D_lm)
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> list[torch.Tensor]:
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        # B,L,H,D with L sequence length, H number of heads, D head dim
        # concatenate on the number of embeddings/tokens
        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        seq_len = query_states.shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        attention_mask_ = _attention_mask
        position_ids_ = _position_ids

        query_states = apply_rope(query_states, position_ids_)
        key_states = apply_rope(key_states, position_ids_)

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)

        attention_interface = self.get_attention_interface()

        att_output = attention_interface(
            attention_mask_, batch_size, head_dim, query_states, key_states, value_states
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> list[torch.Tensor]:
        attention_interface = self.get_attention_interface()

        att_outputs = []
        assert len(inputs_embeds) == 2 or (use_cache and past_key_values is not None and not fill_kv_cache), (
            f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"
        )

        if len(inputs_embeds) == 2 and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]

            hidden_states = layer.input_layernorm(inputs_embeds[0])

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)

            att_output = attention_interface(
                prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_outputs.append(att_output)
        else:
            expert_position_id = position_ids

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        # Expert
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])

            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *key_states.shape[:2], -1
            )
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )  # k_proj should have same dim as kv

            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *value_states.shape[:2], -1
            )
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            expert_position_id = (
                expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            )  # start from 0
            expert_attention_mask = attention_mask[
                :, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :
            ]  # take into account kv

            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            att_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        # att_output = att_output.to(dtype=models[i].dtype)
        return att_outputs, past_key_values

    def get_model_layers(self, models: list) -> list:
        vlm_layers = []
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = models[1].layers[expert_layer_index]
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        depth_injection_fn=None,
    ):
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        # RMSNorm
        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = (
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                )  # in case of self_attn
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual

                    # PointVLA-style depth injection into action expert (stream i==1)
                    if i == 1 and depth_injection_fn is not None:
                        delta = depth_injection_fn(layer_idx)
                        if delta is not None:
                            out_emb = out_emb + delta

                    outputs_embeds.append(out_emb)

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        attention_interface = self.eager_attention_forward
        return attention_interface

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min  # -2.3819763e38  # See gemma/modules.py
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
