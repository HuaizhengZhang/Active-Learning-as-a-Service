#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: May 23, 2022
"""
from pydantic import AnyHttpUrl, BaseModel, BaseSettings, PositiveInt

from alaas.types.models.al_worker import ALWorkerConfigUnion


class ALServerConfig(BaseModel):
    url: AnyHttpUrl
    worker: ALWorkerConfigUnion


class ALConfig(BaseModel):
    budget: PositiveInt
    strategy: ...
    al_server: ALServerConfig


class Config(BaseSettings):
    name: str
    version: str
    active_learning: ALConfig
