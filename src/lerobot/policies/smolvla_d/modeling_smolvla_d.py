#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""SmolVLA-D: SmolVLA + PointVLA-style stereo depth injection into the action expert."""

import math
from collections import deque
from typing import Callable, TypedDict, Unpack

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla_d.configuration_smolvla_d import SmolVLADConfig
from lerobot.policies.smolvla_d.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import get_safe_dtype


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


# ---------------------------------------------------------------------------
# Positional / utility helpers (identical to vanilla SmolVLA-D)
# ---------------------------------------------------------------------------

def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")
    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    value = unnormalize(value, min_val=0.4, max_val=1.5)
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


def pad_tensor(tensor, max_len, pad_value=0):
    b, d = tensor.shape[:2]
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor
    return padded_tensor


# ---------------------------------------------------------------------------
# PointVLA-style depth injection modules
# ---------------------------------------------------------------------------

class PointCloudEncoder(nn.Module):
    """
    Hierarchical Conv1d encoder for point clouds, following PointVLA (Fig. 2).

    Architecture:
        Input: (B, N, 3) — batched point cloud with N sampled 3-D points.
        4 blocks of Conv1d → LeakyReLU, each followed by a global max-pool.
        Features from all blocks are concatenated (multi-scale) and projected
        to depth_embed_dim via a final Linear layer.
        Output: (B, depth_embed_dim)
    """

    def __init__(self, in_channels: int = 3, channels: tuple[int, ...] = (64, 128, 256, 512), out_dim: int = 128):
        super().__init__()
        self.convs = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(prev, ch, kernel_size=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )
            )
            prev = ch
        self.out_proj = nn.Linear(sum(channels), out_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, 3) → (B, 3, N) for Conv1d
        x = x.transpose(1, 2)
        features = []
        for conv in self.convs:
            x = conv(x)
            features.append(x.max(dim=2).values)  # global max-pool → (B, ch)
        out = torch.cat(features, dim=1)           # (B, sum_channels)
        return self.out_proj(out)                   # (B, out_dim)


class DepthFeatureAdapter(nn.Module):
    """
    Per-layer adapter that projects depth embeddings into expert hidden space.

    Follows PointVLA's zero-initialised injection:
        delta = zero_linear( SiLU( linear( depth_emb ) ) )

    The zero-init on `zero_linear` guarantees that at initialisation the
    adapter contributes nothing — the model starts identical to vanilla
    SmolVLA-D and learns to exploit depth progressively.
    """

    def __init__(self, depth_dim: int, expert_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(depth_dim, depth_dim),
            nn.SiLU(),
            nn.Linear(depth_dim, depth_dim),
        )
        self.zero_proj = nn.Linear(depth_dim, expert_dim, bias=False)
        nn.init.zeros_(self.zero_proj.weight)  # zero-init: no effect at start

    def forward(self, depth_emb: Tensor) -> Tensor:
        # depth_emb: (B, depth_dim) → (B, expert_dim)
        return self.zero_proj(self.mlp(depth_emb))


class DepthInjector(nn.Module):
    """One DepthFeatureAdapter per injection layer, keyed by layer index."""

    def __init__(self, injection_layers: list[int], depth_dim: int, expert_dim: int):
        super().__init__()
        self.adapters = nn.ModuleDict(
            {str(l): DepthFeatureAdapter(depth_dim, expert_dim) for l in injection_layers}
        )

    def make_injection_fn(self, depth_emb: Tensor) -> Callable[[int], Tensor | None]:
        """Return a closure used as `depth_injection_fn` in SmolVLMWithExpertModel.forward()."""
        def _fn(layer_idx: int) -> Tensor | None:
            key = str(layer_idx)
            if key not in self.adapters:
                return None
            delta = self.adapters[key](depth_emb)  # (B, expert_dim)
            return delta.unsqueeze(1)               # (B, 1, expert_dim) — broadcasts over L
        return _fn


# ---------------------------------------------------------------------------
# Main policy class
# ---------------------------------------------------------------------------

class SmolVLADPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLADConfig
    name = "smolvla_d"

    def __init__(self, config: SmolVLADConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        # Async stereo depth worker — None by default, activated by setup_stereo_depth()
        self.depth_worker = None
        self.reset()

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def init_rtc_processor(self):
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    # ------------------------------------------------------------------
    # Stereo depth API (inference)
    # ------------------------------------------------------------------

    def setup_stereo_depth(self, calib_path: str, **processor_kwargs):
        """
        Enable real-time stereo depth during inference.

        Call this once before starting the inference loop:
            policy.setup_stereo_depth("camera/stereo_calibration_result.npz")

        Then feed new frames each step with update_stereo_frame().
        """
        from lerobot.policies.smolvla_d.depth_processor_smolvla_d import (
            StereoDepthProcessor,
            StereoDepthWorker,
        )
        proc = StereoDepthProcessor(calib_path, **processor_kwargs)
        self.depth_worker = StereoDepthWorker(proc)

    def update_stereo_frame(self, stereo_frame_bgr: np.ndarray):
        """
        Submit a new 2560×800 BGR stereo frame to the background depth worker.
        Non-blocking — result becomes available asynchronously.
        """
        if self.depth_worker is not None:
            self.depth_worker.submit(stereo_frame_bgr)

    def _get_depth_point_cloud(self, batch: dict[str, Tensor]) -> Tensor | None:
        """
        Returns a (B, K, 3) float32 tensor of sampled depth points, or None.

        Priority:
          1. Pre-computed tensor from batch["depth_point_cloud"] (training / offline)
          2. Cached result from the async StereoDepthWorker (live inference)
        """
        if "depth_point_cloud" in batch:
            return batch["depth_point_cloud"]

        if self.depth_worker is not None:
            pts = self.depth_worker.get_latest_points()
            if pts is not None and len(pts) > 0:
                k = self.config.depth_num_sample_points
                n = len(pts)
                if n >= k:
                    idx = np.random.choice(n, k, replace=False)
                else:
                    idx = np.random.choice(n, k, replace=True)
                sampled = pts[idx].astype(np.float32)                      # (K, 3)
                device = next(self.parameters()).device
                return torch.from_numpy(sampled).unsqueeze(0).to(device)   # (1, K, 3)

        return None

    # ------------------------------------------------------------------
    # Optimiser / PEFT helpers
    # ------------------------------------------------------------------

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_default_peft_targets(self) -> dict[str, any]:
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        # Also include depth encoder and injector adapter weights
        depth_modules = "point_cloud_encoder|depth_injector"
        target_modules = (
            rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj"
            rf"|model\.({common_projections})"
            rf"|model\.({depth_modules}))"
        )
        return {"target_modules": target_modules, "modules_to_save": []}

    def _validate_peft_config(self, peft_config) -> None:
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging
            logging.warning(
                "Training SmolVLA-D from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        point_cloud = self._get_depth_point_cloud(batch)
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=noise, point_cloud=point_cloud, **kwargs
        )
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)
        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        assert not self._rtc_enabled(), "RTC is not supported for select_action, use it with predict_action_chunk"
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        # Depth point cloud is optional: provided when dataset contains pre-computed depth
        point_cloud = batch.get("depth_point_cloud")
        loss_dict = {}
        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time,
            point_cloud=point_cloud,
        )
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()
        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    # ------------------------------------------------------------------
    # Image / state preprocessing
    # ------------------------------------------------------------------

    def prepare_images(self, batch):
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            # Stereo cameras: split side-by-side frame, take the RIGHT half for the VLM.
            # The full frame is used for depth computation separately.
            if key in self.config.stereo_camera_keys:
                mid = img.shape[-1] // 2
                img = img[..., mid:]   # (B, C, H, W//2)
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0
            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


# ---------------------------------------------------------------------------
# Core flow-matching model
# ---------------------------------------------------------------------------

class VLAFlowMatching(nn.Module):
    def __init__(self, config: SmolVLADConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        expert_hidden = self.vlm_with_expert.expert_hidden_size
        vlm_hidden = self.vlm_with_expert.config.text_config.hidden_size

        self.state_proj = nn.Linear(self.config.max_state_dim, vlm_hidden)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, self.config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)

        # ------------------------------------------------------------------
        # PointVLA depth injection modules
        # ------------------------------------------------------------------
        self.point_cloud_encoder = PointCloudEncoder(
            in_channels=3,
            channels=tuple(config.depth_point_encoder_channels),
            out_dim=config.depth_embed_dim,
        )
        self.depth_injector = DepthInjector(
            injection_layers=config.depth_injection_layers,
            depth_dim=config.depth_embed_dim,
            expert_dim=expert_hidden,
        )

        self.set_requires_grad()

        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj
        # Depth modules are always trainable (they're new, not part of the frozen expert)
        for params in self.point_cloud_encoder.parameters():
            params.requires_grad = True
        for params in self.depth_injector.parameters():
            params.requires_grad = True

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        return time_beta * 0.999 + 0.001

    # ------------------------------------------------------------------
    # Depth helpers
    # ------------------------------------------------------------------

    def _encode_point_cloud(self, point_cloud: Tensor | None) -> Tensor | None:
        """
        Encode a (B, N, 3) point cloud to (B, depth_embed_dim) via PointCloudEncoder.
        Returns None if point_cloud is None.
        """
        if point_cloud is None:
            return None
        depth_emb = self.point_cloud_encoder(point_cloud)

        # Dropout during training: zero-out full samples with prob depth_dropout_prob.
        # This keeps the model robust when depth is unavailable at inference time.
        if self.training and self.config.depth_dropout_prob > 0:
            keep = (
                torch.rand(depth_emb.shape[0], 1, device=depth_emb.device)
                > self.config.depth_dropout_prob
            ).to(depth_emb.dtype)
            depth_emb = depth_emb * keep

        return depth_emb

    def _make_injection_fn(self, depth_emb: Tensor | None):
        """Build the depth_injection_fn closure to pass to SmolVLMWithExpertModel.forward()."""
        if depth_emb is None:
            return None
        return self.depth_injector.make_injection_fn(depth_emb)

    # ------------------------------------------------------------------
    # Prefix / suffix embedding
    # ------------------------------------------------------------------

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None):
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)
            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device
        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1] * states_seq_len
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]
        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)
        att_masks = att_masks.expand(bsize, -1)
        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.vlm_with_expert.expert_hidden_size,
            self.config.min_period, self.config.max_period, device=device,
        )
        time_emb = time_emb.type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        point_cloud: Tensor | None = None,
    ) -> Tensor:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Encode depth once; None → no injection (vanilla training without depth data)
        depth_emb = self._encode_point_cloud(point_cloud)
        injection_fn = self._make_injection_fn(depth_emb)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            depth_injection_fn=injection_fn,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    # ------------------------------------------------------------------
    # Inference (KV-cached denoising)
    # ------------------------------------------------------------------

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        point_cloud: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Encode depth once for all denoising steps
        depth_emb = self._encode_point_cloud(point_cloud)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Prefix pass: expert stream is None → no depth injection needed
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            depth_injection_fn=None,
        )

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                    depth_emb=depth_emb,
                )

            if self._rtc_enabled():
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                    inference_delay=kwargs.get("inference_delay"),
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            else:
                v_t = denoise_step_partial_call(x_t)
            x_t = x_t + dt * v_t
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        depth_emb: Tensor | None = None,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        injection_fn = self._make_injection_fn(depth_emb)

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            depth_injection_fn=injection_fn,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
