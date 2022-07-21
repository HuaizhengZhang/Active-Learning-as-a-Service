import abc
from typing import List, Any, Dict
from pydantic import Field, PositiveInt, BaseModel


class ALModelBase(BaseModel):
    name: str
    batch_size: PositiveInt
    hub: str
    model: str
    device: str

    args: List[str] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field({'pretrained': True})
