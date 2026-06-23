import types

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla_layercut.configuration_smolvla_layercut import SmolVLALayercutConfig


class SmolVLALayercutPolicy(SmolVLAPolicy):
    config_class = SmolVLALayercutConfig  # type: ignore[assignment]
    name = "smolvla_layercut"  # type: ignore[assignment]

    def __init__(self, config, ablate_layer_indices=None, ablate_layer_range=None, **kwargs):
        super().__init__(config, **kwargs)

        # Los argumentos directos tienen prioridad sobre el config
        if ablate_layer_indices is not None:
            self.config.ablate_layer_indices = ablate_layer_indices
        if ablate_layer_range is not None:
            self.config.ablate_layer_range = ablate_layer_range

        indices = self._resolve_ablate_indices()
        if indices:
            self._apply_ablation(indices)

    def _resolve_ablate_indices(self) -> list[int]:
        """Fusiona ablate_layer_indices y ablate_layer_range en una lista ordenada sin duplicados."""
        indices = set(self.config.ablate_layer_indices or [])
        r = self.config.ablate_layer_range
        if r is not None:
            if len(r) != 2:
                raise ValueError(f"ablate_layer_range debe ser [start, end], recibido: {r}")
            indices |= set(range(r[0], r[1] + 1))  # intervalo cerrado [start, end]
        return sorted(indices)

    def _apply_ablation(self, ablate_indices: list[int]):
        """
        Parchea get_model_layers y forward_attn_layer del SmolVLMWithExpertModel para
        que las capas del action expert en ablate_indices actúen como identidad (passthrough),
        sin alterar en ningún caso las capas del VLM.

        Casos cubiertos:
          - forward de entrenamiento: inputs_embeds = [prefix, suffix]
          - fill_kv_cache (inferencia, primer paso): inputs_embeds = [prefix, suffix]
          - denoising step (inferencia, pasos N→0): inputs_embeds = [None, suffix]
            → sin el patch, forward_attn_layer hace torch.cat([]) y lanza RuntimeError
              cuando la capa es par (self-attn) y el expert está ablacionado.
        """
        vlm_with_expert = self.model.vlm_with_expert
        original_get_model_layers = vlm_with_expert.get_model_layers
        original_forward_attn_layer = vlm_with_expert.forward_attn_layer

        ablated_set = set(ablate_indices)

        def patched_get_model_layers(self_inner, models):
            vlm_layers, expert_layers = original_get_model_layers(models)

            # Conjunto de objetos nn.Module a reemplazar por None
            n_expert = len(self_inner.lm_expert.layers)
            layers_to_null = {
                self_inner.lm_expert.layers[i]
                for i in ablated_set
                if i < n_expert
            }

            new_expert = [
                None if (layer is not None and layer in layers_to_null) else layer
                for layer in expert_layers
            ]
            return [vlm_layers, new_expert]

        def patched_forward_attn_layer(
            self_inner,
            model_layers,
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            batch_size,
            head_dim,
            use_cache=True,
            fill_kv_cache=True,
            past_key_values=None,
        ):
            # Si para este layer_idx ningún stream tiene capa activa Y hidden_states,
            # devolvemos identidad sin llamar a torch.cat (evita RuntimeError).
            has_active = any(
                h is not None and model_layers[i][layer_idx] is not None
                for i, h in enumerate(inputs_embeds)
            )
            if not has_active:
                # Devolvemos un output None por cada stream con datos para que el
                # bucle de outputs en forward() haga el passthrough de identidad.
                identity_out = [None if h is None else None for h in inputs_embeds]
                return identity_out, past_key_values

            return original_forward_attn_layer(
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

        vlm_with_expert.get_model_layers = types.MethodType(
            patched_get_model_layers, vlm_with_expert
        )
        vlm_with_expert.forward_attn_layer = types.MethodType(
            patched_forward_attn_layer, vlm_with_expert
        )
