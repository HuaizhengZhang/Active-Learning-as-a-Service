"""
Diversity based active learning methods.
@author huangyz0918 (huangyz0918@gmail.com)
@date 02/08/2022
"""

import numpy as np
from .base import Strategy


class BadgeSampling(Strategy):
    """
    TODO: to build.
    Reference: Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (https://arxiv.org/abs/1906.03671).
    @inproceedings{ash2019deep,
      author    = {Jordan T. Ash and
                   Chicheng Zhang and
                   Akshay Krishnamurthy and
                   John Langford and
                   Alekh Agarwal},
      title     = {Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds},
      booktitle = {8th International Conference on Learning Representations, {ICLR} 2020,
                   Addis Ababa, Ethiopia, April 26-30, 2020},
      publisher = {OpenReview.net},
      year      = {2020}
    }
    """

    def __init__(self, path_mapping, n_drop):
        super(BadgeSampling, self).__init__(path_mapping, n_drop)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        if self.n_drop:
            embeddings = embeddings.mean(0)
        _uncertainties = np.amax(embeddings, axis=1)
        return _path_list[_uncertainties.argsort()[:int(n)]]
