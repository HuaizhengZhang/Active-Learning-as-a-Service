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

    def __init__(self, infer_func, proc_func, model_name, batch_size, address='localhost:8900'):
        super(KMeansSampling, self).__init__(infer_func=infer_func, proc_func=proc_func)
        self.model_name = model_name
        self.batch_size = batch_size
        self.address = address

    def query(self, n):
        self.check_query_num(n)
        path_list = np.array([x[2] for x in self.data_pool])

        # update the inference results.
        while len(self.data_not_inferred) > 0:
            md5_list, uuid_list, data_list = self.proc_func(self.data_not_inferred)
            features = self.infer_func(data_list, self.batch_size,
                                       model_name=self.model_name, address=self.address)
            # TODO: data representations are usually larger than original data size.
            for i in range(len(features)):
                self.db_manager.update_inference(uuid_list[i], features[i])
            self.data_not_inferred = self.db_manager.get_rows(inferred=False)

        kmeans = KMeans(n_clusters=n)
        kmeans.fit(features)
        distance = kmeans.transform(features)
        query_ids = np.argmin(distance, axis=0)
        return path_list[query_ids]
