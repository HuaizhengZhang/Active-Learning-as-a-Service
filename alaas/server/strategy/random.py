"""
Random sampling methods.
@author huangyz0918 (huangyz0918@gmail.com)
@date 06/05/2022
"""

import numpy as np
from .base import Strategy


class RandomSampling(Strategy):
    """
    Randomly Selected the query samples.
    """

    def __init__(self, path_mapping, n_drop):
        super(RandomSampling, self).__init__(path_mapping, n_drop)

    def query(self, n, embedding=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)
        return _path_list[np.random.choice(self.pool_size, int(n), replace=False)]
