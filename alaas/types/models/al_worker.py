#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: May 23, 2022
"""
import abc
from enum import Enum
from typing import Union

from pydantic import AnyUrl, BaseModel, validator


class ALWorkerType(Enum):
    """Enum of supported AL Worker"""
    DOCKER = 'docker'


class ALWorkerConfigBase(BaseModel, abc.ABC):
    url: AnyUrl = AnyUrl('localhost', scheme='')
    type: ALWorkerType

    __required_type__: ALWorkerType

    @validator('type')
    def check_layer_type(cls, worker_type: ALWorkerType) -> ALWorkerType:
        """
        Checks worker type value provided is the same as the required value.
        This is to generate validator for check :code:`worker_type` field of subclasses of :class:`ALWorkerType`.
        """
        if worker_type != cls.__required_type__:
            raise ValueError(f'Expected {cls.__required_type__} but got {worker_type}')
        return worker_type


# TODO: add more docker worker support, currently is hard-coded for Triton Docker.
class TritonDockerConfig(ALWorkerConfigBase):
    docker_repo: str = 'nvcr.io/nvidia/tritonserver'
    tag: str

    __required_type__ = ALWorkerType.DOCKER


ALWorkerConfigUnion = Union[
    TritonDockerConfig
]
