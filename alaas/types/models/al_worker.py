#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: May 23, 2022
"""
import re
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Union, Dict, Any

from pydantic import AnyUrl, root_validator, Extra, Field, validator

from alaas.types.models.utils import TypeCheckMixin


class ALWorkerType(Enum):
    """Enum of supported AL Worker"""
    DOCKER = 'docker'
    UNKNOWN = 'unknown'


class ALWorkerConfigBase(TypeCheckMixin[ALWorkerType], ABC, extra=Extra.ignore):
    url: AnyUrl = AnyUrl('localhost', scheme='')


class TritonDockerVariationType(Enum):
    PY3 = 'py3'
    PY3_SDK = 'py3-sdk'
    PY3_MIN = 'py3-min'
    PYT_PYTHON_PY3 = 'pyt-python-py3'
    TF2_PYTHON_PY3 = 'tf2-python-py3'


# TODO: add more docker worker support, currently is hard-coded for Triton Docker.
class TritonDockerConfig(ALWorkerConfigBase):
    docker_repo: str = 'nvcr.io/nvidia/tritonserver'
    tag: str = None
    version: str = Field(None, regex=r'^[0-9]{2}\.[0-9]{2}$')
    variation: TritonDockerVariationType = None
    gpus: str = 'all'
    model_repository_path: Path = Field(Path.home() / '.alaas/models')
    docker_kwargs: Dict[str, Any] = Field(default_factory=dict)
    command: Dict[str, str] = Field(default_factory=dict)

    __required_type__ = ALWorkerType.DOCKER

    def __init__(self, **data):
        super().__init__(**data)
        # create model repository directory
        self.model_repository_path.mkdir(exist_ok=True, parents=True)

    @root_validator(pre=True)
    def check_tag(cls, values):
        tag = values.get('tag', None)
        version, variation = values.get('version', None), values.get('variation', None)
        if bool(tag) == bool(version and variation):
            raise ValueError(
                'One and only one set of fields (`tag`) and (`version`, `variation`)'
                'should be set.'
            )
        if tag is not None:
            tag_pattern = re.compile(r'(.+?)-(.+)')
            matches = tag_pattern.match(tag)
            if matches:
                values['version'], values['variation'] = matches[1], matches[2]
            else:
                values['version'], values['variation'] = None, None
        else:
            values['tag'] = f'{version}-{variation}'

        return values


class OthersConfig(ALWorkerConfigBase, ):
    __required_type__ = ALWorkerType.UNKNOWN

    @root_validator
    def raise_fail(cls, values):
        raise ValueError('Unrecognized type')


ALWorkerConfigUnion = Union[
    TritonDockerConfig, OthersConfig
]
