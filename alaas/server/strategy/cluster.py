"""
Clustering based active learning methods.
@author huangyz0918 (huangyz0918@gmail.com)
@date 18/07/2022
"""

import numpy as np
from sklearn.cluster import KMeans

from .base import Strategy


class KMeansSampling(Strategy):
    """
    KMeans Cluster Sampling Method.
    """

    def __init__(self, pool_size, path_mapping):
        super(KMeansSampling, self).__init__(pool_size, path_mapping)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)

        kmeans = KMeans(n_clusters=n)
        kmeans.fit(embeddings)
        distance = kmeans.transform(embeddings)
        query_ids = np.argmin(distance, axis=0)
        return _path_list[query_ids]
