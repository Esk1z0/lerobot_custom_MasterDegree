from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

@PreTrainedConfig.register_subclass("smolvla_layercut")
@dataclass
class SmolVLALayercutConfig(SmolVLAConfig):
    # Lista de indices de las capas del action expert que queremos bypassear
    ablate_layer_indices: list[int] = field(default_factory=list)
