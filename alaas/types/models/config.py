#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Author: huangyz0918
Email: yuanmingleee@gmail.com
Email: huangyz0918@gmail.com
Date: May 23, 2022
"""
from pydantic import BaseModel, BaseSettings

from alaas.types.models.al_strategy import ALStrategyConfigUnion


class ALWorkerConfig(BaseModel):
    protocol: str
    host: str
    port: int
    replicas: int


class ALConfig(BaseModel):
    strategy: ALStrategyConfigUnion
    al_worker: ALWorkerConfig


class Config(BaseSettings):
    name: str
    version: str
    active_learning: ALConfig
