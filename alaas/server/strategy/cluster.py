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

    def __init__(self, path_mapping, n_drop):
        super(KMeansSampling, self).__init__(path_mapping, n_drop)

    def query(self, n, embeddings=None):
        self.check_query_num(n)
        _path_list = np.array(self.path_mapping)

        if self.n_drop:
            embeddings = embeddings.mean(0)

        kmeans = KMeans(n_clusters=n)
        kmeans.fit(embeddings)
        distance = kmeans.transform(embeddings)

        query_ids = np.argmin(distance, axis=0)
        return _path_list[query_ids]
