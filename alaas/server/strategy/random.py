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

    def __init__(self):
        super(RandomSampling, self).__init__(None, None)

    def query(self, n):
        self.check_query_num(n)
        path_list = np.array([x[2] for x in self.data_pool])
        return path_list[np.random.choice(len(self.data_pool), n, replace=False)]
