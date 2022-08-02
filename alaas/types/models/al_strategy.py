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

from alaas.types.models.al_model import ALModelBase
from alaas.types.models.utils import TypeCheckMixin


class ALStrategyType(Enum):
    """Enum of supported AL strategy"""
    RANDOM_SAMPLING = 'RandomSampling'
    LEAST_CONFIDENCE = 'LeastConfidence'
    MARGIN_CONFIDENCE = 'MarginConfidence'
    RATIO_CONFIDENCE = 'RatioConfidence'
    ENTROPY_SAMPLING = 'EntropySampling'
    KMEANS_SAMPLING = 'KMeansSampling'
    BADGE_SAMPLING = 'BadgeSampling'
    BAYESIAN_DISAGREEMENT = 'BayesianDisagreement'
    MEAN_STD = 'MeanSTDSampling'
    VAR_RATIO = 'VarRatioSampling'


class ALStrategyBase(TypeCheckMixin[ALStrategyType], ABC):
    pass


class RandomSamplingConfig(ALStrategyBase):
    model: ALModelBase

    __required_type__ = ALStrategyType.RANDOM_SAMPLING


class LeastConfidenceConfig(ALStrategyBase):
    model: ALModelBase

    __required_type__ = ALStrategyType.LEAST_CONFIDENCE


class MarginConfidenceConfig(ALStrategyBase):
    model: ALModelBase

    __required_type__ = ALStrategyType.MARGIN_CONFIDENCE


class RatioConfidenceConfig(ALStrategyBase):
    model: ALModelBase

    __required_type__ = ALStrategyType.RATIO_CONFIDENCE


class EntropySamplingConfig(ALStrategyBase):
    model: ALModelBase

    __required_type__ = ALStrategyType.ENTROPY_SAMPLING


class KMeansSamplingConfig(ALStrategyBase):
    model: ALModelBase

    __required_type__ = ALStrategyType.KMEANS_SAMPLING


class BayesianDisagreementConfig(ALStrategyBase):
    model: ALModelBase

    __required_type__ = ALStrategyType.BAYESIAN_DISAGREEMENT


class BadgeSamplingConfig(ALStrategyBase):
    model: ALModelBase

    __required_type__ = ALStrategyType.BADGE_SAMPLING


class MeanSTDSamplingConfig(ALStrategyBase):
    model: ALModelBase

    __required_type__ = ALStrategyType.MEAN_STD


class VarRatioSamplingConfig(ALStrategyBase):
    model: ALModelBase

    __required_type__ = ALStrategyType.VAR_RATIO


ALStrategyConfigUnion = Union[
    RandomSamplingConfig,
    LeastConfidenceConfig,
    MarginConfidenceConfig,
    RatioConfidenceConfig,
    EntropySamplingConfig,
    KMeansSamplingConfig,
    BayesianDisagreementConfig,
    BadgeSamplingConfig,
    MeanSTDSamplingConfig,
    VarRatioSamplingConfig
]
