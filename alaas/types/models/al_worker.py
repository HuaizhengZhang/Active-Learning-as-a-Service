#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: May 23, 2022
"""
from abc import ABC
from enum import Enum
from typing import Union

from pydantic import AnyUrl, root_validator, Extra

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

    __required_type__ = ALWorkerType.DOCKER


class OthersConfig(ALWorkerConfigBase, ):
    __required_type__ = ALWorkerType.UNKNOWN

    @root_validator
    def raise_fail(cls, values):
        raise ValueError('Unrecognized type')


ALWorkerConfigUnion = Union[
    TritonDockerConfig, OthersConfig
]
