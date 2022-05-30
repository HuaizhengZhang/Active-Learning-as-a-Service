#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: May 23, 2022
"""
from pydantic import BaseModel, BaseSettings, PositiveInt

from alaas.types.models.al_strategy import ALStrategyConfigUnion
from alaas.types.models.al_worker import ALWorkerConfigUnion


class ALServerConfig(BaseModel):
    url: str
    worker: ALWorkerConfigUnion


class ALConfig(BaseModel):
    strategy: ALStrategyConfigUnion
    al_server: ALServerConfig


class Config(BaseSettings):
    name: str
    version: str
    active_learning: ALConfig


if __name__ == '__main__':
    import yaml

    with open('../../../examples/resnet_triton.yml') as f:
        obj = yaml.safe_load(f)
    print(Config.parse_obj(obj))
