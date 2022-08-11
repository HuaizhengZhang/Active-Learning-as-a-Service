"""
Uncertainty based active learning methods.
@author huangyz0918 (huangyz0918@gmail.com)
@date 28/05/2022
"""

import numpy as np
from .base import Strategy


class LeastConfidence(Strategy):
    """
    Least Confidence Sampling.
    Reference: A Sequential Algorithm for Training Text Classifiers (1994).
    (https://arxiv.org/pdf/cmp-lg/9407020)
    """

    def __init__(self, path_mapping, n_drop):
        super(LeastConfidence, self).__init__(path_mapping, n_drop)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        if self.n_drop:
            embeddings = embeddings.mean(0)
        _uncertainties = np.amax(embeddings, axis=1)
        return _path_list[_uncertainties.argsort()[:int(n)]]


class MarginConfidence(Strategy):
    """
    Margin of Confidence Sampling.
    Reference: Active Hidden Markov Models for Information Extraction (2001).
    (https://link.springer.com/chapter/10.1007/3-540-44816-0_31)
    """

    def __init__(self, path_mapping, n_drop):
        super(MarginConfidence, self).__init__(path_mapping, n_drop)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        if self.n_drop:
            embeddings = embeddings.mean(0)
        _probs_sorted = -np.sort(-np.array(embeddings))
        _difference = _probs_sorted[:, 0] - _probs_sorted[:, 1]
        return _path_list[_difference.argsort()[:int(n)]]


class RatioConfidence(Strategy):
    """
    Ratio of Confidence Sampling.
    Reference: Active learning literature survey. (https://minds.wisconsin.edu/handle/1793/60660)
    """

    def __init__(self, path_mapping, n_drop):
        super(RatioConfidence, self).__init__(path_mapping, n_drop)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        if self.n_drop:
            embeddings = embeddings.mean(0)
        _probs_sorted = -np.sort(-np.array(embeddings))
        _difference = _probs_sorted[:, 0] / _probs_sorted[:, 1]
        return _path_list[_difference.argsort()[:int(n)]]


class EntropySampling(Strategy):
    """
    Entropy Sampling.
    Reference: Active learning literature survey. (https://minds.wisconsin.edu/handle/1793/60660)
    """

    def __init__(self, path_mapping, n_drop):
        super(EntropySampling, self).__init__(path_mapping, n_drop)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        if self.n_drop:
            embeddings = embeddings.mean(0)
        _entropy = (embeddings * np.log(embeddings)).sum(1)
        return _path_list[_entropy.argsort()[:int(n)]]


class BayesianDisagreement(Strategy):
    """
    Bayesian Active Learning Disagreement.
    Reference: https://arxiv.org/abs/1703.02910
    """

    def __init__(self, path_mapping, n_drop):
        super(BayesianDisagreement, self).__init__(path_mapping, n_drop)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)

        if self.n_drop:
            embeddings_mean = embeddings.mean(0)
            _entropy = (-embeddings_mean * np.log(embeddings_mean)).sum(1)
            _entropy_mean = (-embeddings * np.log(embeddings)).sum(2).mean(0)

            _uncertainties = _entropy_mean - _entropy
            return _path_list[_uncertainties.argsort()[:int(n)]]
        else:
            raise ValueError("strategy {BayesianDisagreement} requires to set the parameter {n_drop} > 1")


class MeanSTDSampling(Strategy):
    """
    Mean Standard Deviation Sampling.
    Reference: Semantic segmentation of small objects and modeling of uncertainty in urban remote
                sensing images using deep convolutional neural networks, CVPR, 2016
                (https://ieeexplore.ieee.org/document/7789580)
    """

    def __init__(self, path_mapping, n_drop):
        super(MeanSTDSampling, self).__init__(path_mapping, n_drop)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)

        if self.n_drop:
            sigma_c = np.std(embeddings, axis=0)
            _uncertainties = np.mean(sigma_c, axis=-1)
            _uncertainties_sorted = np.argsort(-np.array(_uncertainties))
            return _path_list[_uncertainties_sorted[:int(n)]]
        else:
            raise ValueError("strategy {BayesianDisagreement} requires to set the parameter {n_drop} > 1")


class VarRatioSampling(Strategy):
    """
    Variation Ratios Sampling.
    Reference: Elementary applied statistics: for students in behavioral science. New York: Wiley, 1965
    """

    def __init__(self, path_mapping, n_drop):
        super(VarRatioSampling, self).__init__(path_mapping, n_drop)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)

        if self.n_drop:
            embeddings = embeddings.mean(0)

        _preds = np.max(embeddings, axis=1)
        _uncertainties = np.argsort(-np.array(1.0 - _preds))
        return _path_list[_uncertainties[:int(n)]]
