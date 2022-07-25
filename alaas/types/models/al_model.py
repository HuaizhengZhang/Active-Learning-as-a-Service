from enum import Enum
from typing import List, Any, Dict, Optional
from pydantic import Field, PositiveInt, BaseModel


class ModalityType(Enum):
    IMAGE = 'MODALITY_IMAGE'
    TEXT = 'MODALITY_TEXT'


class ALModelBase(BaseModel):
    name: str
    batch_size: PositiveInt
    hub: str
    device: str
    tokenizer: Optional[str] = None
    task: Optional[str] = None

    args: List[str] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field({'pretrained': True})
