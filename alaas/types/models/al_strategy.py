#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Author: huangyz0918 (huangyz0918@gmail.com)
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
    RANDOM_SAMPLING = 'RandomSampling'
    LEAST_CONFIDENCE = 'LeastConfidence'
    MARGIN_CONFIDENCE = 'MarginConfidence'
    RATIO_CONFIDENCE = 'RatioConfidence'
    ENTROPY_SAMPLING = 'EntropySampling'


class ALStrategyBase(TypeCheckMixin[ALStrategyType], ABC):
    pass


class RandomSamplingConfig(ALStrategyBase):
    infer_model: InferModelConfigUnion

    __required_type__ = ALStrategyType.RANDOM_SAMPLING


class LeastConfidenceConfig(ALStrategyBase):
    infer_model: InferModelConfigUnion

    __required_type__ = ALStrategyType.LEAST_CONFIDENCE


class MarginConfidenceConfig(ALStrategyBase):
    infer_model: InferModelConfigUnion

    __required_type__ = ALStrategyType.MARGIN_CONFIDENCE


class RatioConfidenceConfig(ALStrategyBase):
    infer_model: InferModelConfigUnion

    __required_type__ = ALStrategyType.RATIO_CONFIDENCE


class EntropySamplingConfig(ALStrategyBase):
    infer_model: InferModelConfigUnion

    __required_type__ = ALStrategyType.ENTROPY_SAMPLING


ALStrategyConfigUnion = Union[
    RandomSamplingConfig,
    LeastConfidenceConfig,
    MarginConfidenceConfig,
    RatioConfidenceConfig,
    EntropySamplingConfig
]
