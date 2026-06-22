import types

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla_layercut.configuration_smolvla_layercut import SmolVLALayercutConfig

class SmolVLALayercutPolicy(SmolVLAPolicy):
    config_class = SmolVLALayercutConfig
    name = "smolvla_layercut"

    def __init__(self, config, ablate_layer_indices=None, **kwargs):
        super().__init__(config, **kwargs)
        
        # Priority to the argument passed directly
        if ablate_layer_indices is not None:
            self.config.ablate_layer_indices = ablate_layer_indices
            
        # Apply the monkeypatch if there are layers to ablate
        ablate_indices = getattr(self.config, "ablate_layer_indices", None)
        if ablate_indices:
            self._apply_ablation(ablate_indices)
            
    def _apply_ablation(self, ablate_layer_indices):
        vlm_with_expert = self.model.vlm_with_expert
        original_get_model_layers = vlm_with_expert.get_model_layers
        original_forward_attn_layer = vlm_with_expert.forward_attn_layer
        
        def ablated_get_model_layers(self_expert, models):
            vlm_layers, expert_layers = original_get_model_layers(models)
            target_layers = [self_expert.lm_expert.layers[i] for i in ablate_layer_indices]
            
            new_expert_layers = []
            for layer in expert_layers:
                if layer in target_layers:
                    new_expert_layers.append(None)
                else:
                    new_expert_layers.append(layer)
            return [vlm_layers, new_expert_layers]
            
        def ablated_forward_attn_layer(self_expert, *args, **kwargs):
            try:
                return original_forward_attn_layer(*args, **kwargs)
            except ValueError as e:
                if "expected a non-empty list of Tensors" in str(e):
                    past_key_values = kwargs.get("past_key_values")
                    if past_key_values is None and len(args) >= 10:
                        past_key_values = args[9]
                    return [None], past_key_values
                raise e
                
        vlm_with_expert.get_model_layers = types.MethodType(ablated_get_model_layers, vlm_with_expert)
        vlm_with_expert.forward_attn_layer = types.MethodType(ablated_forward_attn_layer, vlm_with_expert)
