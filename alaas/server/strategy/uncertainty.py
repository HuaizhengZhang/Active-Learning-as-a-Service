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

    def __init__(self, pool_size, path_mapping, n_drop):
        super(LeastConfidence, self).__init__(pool_size, path_mapping, n_drop)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        if self.n_drop:
            _uncertainties = np.amax(embeddings.mean(0), axis=1)
        else:
            _uncertainties = np.amax(embeddings, axis=1)
        return _path_list[_uncertainties.argsort()[:int(n)]]


class MarginConfidence(Strategy):
    """
    Margin of Confidence Sampling.
    Reference: Active Hidden Markov Models for Information Extraction (2001).
    (https://link.springer.com/chapter/10.1007/3-540-44816-0_31)
    """

    def __init__(self, pool_size, path_mapping, n_drop):
        super(MarginConfidence, self).__init__(pool_size, path_mapping, n_drop)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        if self.n_drop:
            _probs_sorted = -np.sort(-np.array(embeddings.mean(0)))
        else:
            _probs_sorted = -np.sort(-np.array(embeddings))
        _difference = _probs_sorted[:, 0] - _probs_sorted[:, 1]
        return _path_list[_difference.argsort()[:int(n)]]


class RatioConfidence(Strategy):
    """
    Ratio of Confidence Sampling.
    Reference: Active learning literature survey. (https://minds.wisconsin.edu/handle/1793/60660)
    """

    def __init__(self, pool_size, path_mapping, n_drop):
        super(RatioConfidence, self).__init__(pool_size, path_mapping, n_drop)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        if self.n_drop:
            _probs_sorted = -np.sort(-np.array(embeddings.mean(0)))
        else:
            _probs_sorted = -np.sort(-np.array(embeddings))
        _difference = _probs_sorted[:, 0] / _probs_sorted[:, 1]
        return _path_list[_difference.argsort()[:int(n)]]


class EntropySampling(Strategy):
    """
    Entropy Sampling.
    Reference: Active learning literature survey. (https://minds.wisconsin.edu/handle/1793/60660)
    """

    def __init__(self, pool_size, path_mapping, n_drop):
        super(EntropySampling, self).__init__(pool_size, path_mapping, n_drop)

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

    def __init__(self, pool_size, path_mapping, n_drop):
        super(BayesianDisagreement, self).__init__(pool_size, path_mapping, n_drop)

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
