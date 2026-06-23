from dataclasses import dataclass, field
from typing import Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvla_layercut")
@dataclass
class SmolVLALayercutConfig(SmolVLAConfig):
    # Índices sueltos del action expert a saltar (ej: [0, 3, 7])
    ablate_layer_indices: list[int] = field(default_factory=list)

    # Intervalo cerrado [start, end] de capas del action expert a saltar (ej: [4, 8])
    # Se combina con ablate_layer_indices si ambos están presentes.
    ablate_layer_range: Optional[list[int]] = None  # [start, end] inclusive
