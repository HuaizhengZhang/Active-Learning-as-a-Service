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

from alaas.types.models.infer_model import InferModelConfigUnion
from alaas.types.models.utils import TypeCheckMixin


class ALStrategyType(Enum):
    """Enum of supported AL strategy"""
    LEAST_CONFIDENCE = 'LeastConfidence'


class ALStrategyBase(TypeCheckMixin[ALStrategyType], ABC):
    pass


class LeastConfidenceConfig(ALStrategyBase):
    infer_model: InferModelConfigUnion

    __required_type__ = ALStrategyType.LEAST_CONFIDENCE


ALStrategyConfigUnion = Union[
    LeastConfidenceConfig,
]
