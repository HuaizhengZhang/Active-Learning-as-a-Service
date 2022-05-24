from abc import ABC
from enum import Enum
from typing import List, Any, Union, Dict

from pydantic import Field, PositiveInt

from alaas.types.models.utils import TypeCheckMixin


class InferModelType(Enum):
    """Enum of supported AL strategy"""
    TORCH_HUB = 'torch_hub'


class InferModelBase(TypeCheckMixin[InferModelType], ABC):
    name: str
    batch_size: PositiveInt
    input: List[Any]
    output: List[Any]
    instance_group: List[Any]


class TorchHubInferModelConfig(InferModelBase):
    hub_name: str
    model: str
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    __required_type__ = InferModelType.TORCH_HUB


InferModelConfigUnion = Union[
    TorchHubInferModelConfig,
]
