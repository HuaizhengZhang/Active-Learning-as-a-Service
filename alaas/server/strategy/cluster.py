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

# class KCenterGreedy(Strategy):
#     """
#     K-Center Greedy Method.
#     """
#
#     def __init__(self, path_mapping, n_drop):
#         super(KCenterGreedy, self).__init__(path_mapping, n_drop)
#
#     def query(self, n, embeddings=None):
#         self.check_query_num(n)
#         _path_list = np.array(self.path_mapping)
#
#         dist_mat = np.matmul(embeddings, embeddings.transpose())
#         sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
#         dist_mat *= -2
#         dist_mat += sq
#         dist_mat += sq.transpose()
#         dist_mat = np.sqrt(dist_mat)
#
#         mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]
#
#         for i in range(n):
#             mat_min = mat.min(axis=1)
#             q_idx_ = mat_min.argmax()
#             q_idx = np.arange(self.pool_size)[~labeled_idxs][q_idx_]
#             labeled_idxs[q_idx] = True
#             mat = np.delete(mat, q_idx_, 0)
#             mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
#
#         return _path_list[np.arange(self.pool_size)[(self.dataset.labeled_idxs ^ labeled_idxs)]]
