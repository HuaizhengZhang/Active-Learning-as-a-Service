"""
    Random sampling methods.
    @author huangyz0918 (huangyz0918@gmail.com)
    @date 06/05/2022
"""
import logging
import warnings

import numpy as np
from .base import Strategy


class RandomSampling(Strategy):
    """
    Randomly Selected the query samples.
    """

    def __init__(self, source_data):
        super(RandomSampling, self).__init__(None, source_data)
        self.source_data = np.array(self.db_manager.read_records())

    def query(self, n):
        data_num = self.source_data.shape[0]
        if n > data_num:
            n = data_num
            warnings.warn("You have query more samples than the current pool size, return all available data.")
        return self.source_data[np.random.choice(data_num, n, replace=False)]
