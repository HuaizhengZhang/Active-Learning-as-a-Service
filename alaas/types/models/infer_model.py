import abc
from enum import Enum
from typing import List, Any, Union, Dict, Optional

from pydantic import Field, PositiveInt, BaseModel, Extra

from alaas.types.models.utils import TypeCheckMixin


class ModelInputOutput(BaseModel, extra=Extra.allow):
    name: Optional[str]
    data_type: str
    dims: List[int]


class InferModelType(Enum):
    """Enum of supported AL strategy"""
    TORCH_HUB = 'torch_hub'


class InferModelBase(TypeCheckMixin[InferModelType], abc.ABC):
    name: str
    batch_size: PositiveInt
    conda_env: str
    input: List[ModelInputOutput]
    output: List[ModelInputOutput]
    instance_group: List[Any] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        # auto generate input/output name as "INPUT0"/"OUTPUT0",
        # "INPUT1"/"OUTPUT1", ...
        # FIXME: this may lead to name collision
        for i, input_shape in enumerate(self.input):
            if not input_shape.name:
                input_shape.name = f'INPUT{i}'
        for i, output_shape in enumerate(self.output):
            if not output_shape.name:
                output_shape.name = f'OUTPUT{i}'

    @property
    @abc.abstractmethod
    def _exclude_keys(self):
        pass

    def to_model_config_dict(self):
        return self.dict(exclude=self._exclude_keys, exclude_none=True, exclude_unset=True)


class TorchHubInferModelConfig(InferModelBase):
    hub_name: str
    model: str
    args: List[str] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field({'pretrained': True})

    __required_type__ = InferModelType.TORCH_HUB

    @property
    def _exclude_keys(self):
        return {
            'type', 'hub_name', 'model', 'kwargs', 'batch_size', 'conda_env',
        }


InferModelConfigUnion = Union[
    TorchHubInferModelConfig,
]
