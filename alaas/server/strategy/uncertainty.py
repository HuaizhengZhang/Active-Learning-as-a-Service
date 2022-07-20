"""
Uncertainty based active learning methods.
@author huangyz0918 (huangyz0918@gmail.com)
@date 28/05/2022
"""

import numpy as np
from .base import Strategy


class LeastConfidence(Strategy):
    """
    Least Confidence Sampling using Triton Inference Server.
    Reference: A Sequential Algorithm for Training Text Classifiers (1994).
    (https://arxiv.org/pdf/cmp-lg/9407020)
    """

    def __init__(self, pool_size, path_mapping):
        super(LeastConfidence, self).__init__(pool_size, path_mapping)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        _uncertainties = np.amax(embeddings, axis=1)
        return _path_list[_uncertainties.argsort()[:int(n)]]


class MarginConfidence(Strategy):
    """
    Margin of Confidence sampling using Triton Inference Server.
    Reference: Active Hidden Markov Models for Information Extraction (2001).
    (https://link.springer.com/chapter/10.1007/3-540-44816-0_31)
    """

    def __init__(self, pool_size, path_mapping):
        super(MarginConfidence, self).__init__(pool_size, path_mapping)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        _probs_sorted = -np.sort(-np.array(embeddings))
        _difference = _probs_sorted[:, 0] - _probs_sorted[:, 1]
        return _path_list[_difference.argsort()[:int(n)]]


class RatioConfidence(Strategy):
    """
    Ratio of Confidence sampling using Triton Inference Server.
    Reference: Active learning literature survey. (https://minds.wisconsin.edu/handle/1793/60660)
    """

    def __init__(self, pool_size, path_mapping):
        super(RatioConfidence, self).__init__(pool_size, path_mapping)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        _probs_sorted = -np.sort(-np.array(embeddings))
        _difference = _probs_sorted[:, 0] / _probs_sorted[:, 1]
        return _path_list[_difference.argsort()[:int(n)]]


class EntropySampling(Strategy):
    """
    Entropy Sampling using Triton Inference Server.
    Reference: Active learning literature survey. (https://minds.wisconsin.edu/handle/1793/60660)
    """

    def __init__(self, pool_size, path_mapping):
        super(EntropySampling, self).__init__(pool_size, path_mapping)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        _entropy = (embeddings * np.log(embeddings)).sum(1)
        return _path_list[_entropy.argsort()[:int(n)]]
