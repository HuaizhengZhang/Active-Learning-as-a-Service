#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: May 23, 2022
"""
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Union, Dict, Any

from pydantic import AnyUrl, root_validator, Extra, Field

from alaas.types.models.utils import TypeCheckMixin


class ALWorkerType(Enum):
    """Enum of supported AL Worker"""
    DOCKER = 'docker'
    UNKNOWN = 'unknown'


class ALWorkerConfigBase(TypeCheckMixin[ALWorkerType], ABC, extra=Extra.ignore):
    url: AnyUrl = AnyUrl('localhost', scheme='')


# TODO: add more docker worker support, currently is hard-coded for Triton Docker.
class TritonDockerConfig(ALWorkerConfigBase):
    docker_repo: str = 'nvcr.io/nvidia/tritonserver'
    tag: str
    gpus: str = 'all'
    model_repository_path: Path = Field(Path.home() / '.alaas/models')
    docker_kwargs: Dict[str, Any] = Field(default_factory=dict)
    command: Dict[str, str] = Field(default_factory=dict)

    __required_type__ = ALWorkerType.DOCKER

    def __init__(self, **data):
        super().__init__(**data)
        # create model repository directory
        self.model_repository_path.mkdir(exist_ok=True, parents=True)


class OthersConfig(ALWorkerConfigBase, ):
    __required_type__ = ALWorkerType.UNKNOWN

    @root_validator
    def raise_fail(cls, values):
        raise ValueError('Unrecognized type')


ALWorkerConfigUnion = Union[
    TritonDockerConfig, OthersConfig
]
