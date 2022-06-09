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

    def __init__(self, infer_func, proc_func, model_name, batch_size, address='localhost:8900'):
        super(LeastConfidence, self).__init__(infer_func=infer_func, proc_func=proc_func)
        self.model_name = model_name
        self.batch_size = batch_size
        self.address = address

    def query(self, n):
        self.check_query_num(n)
        path_list = np.array([x[2] for x in self.data_pool])
        # update the inference results.
        while len(self.data_not_inferred) > 0:
            md5_list, uuid_list, data_list = self.proc_func(self.data_not_inferred)
            probs = self.infer_func(data_list, self.batch_size,
                                    model_name=self.model_name, address=self.address)
            for i in range(len(probs)):
                self.db_manager.update_inference(uuid_list[i], probs[i])
            self.data_not_inferred = self.db_manager.get_rows(inferred=False)
        # query the inference table.
        rows = self.db_manager.get_rows()
        infer_probs = [x[3] for x in rows]
        uncertainties = np.amax(infer_probs, axis=1)
        return path_list[uncertainties.argsort()[:n]]


class MarginConfidence(Strategy):
    """
    Margin of Confidence sampling using Triton Inference Server.
    Reference: Active Hidden Markov Models for Information Extraction (2001).
    (https://link.springer.com/chapter/10.1007/3-540-44816-0_31)
    """

    def __init__(self, infer_func, proc_func, model_name, batch_size, address='localhost:8900'):
        super(MarginConfidence, self).__init__(infer_func=infer_func, proc_func=proc_func)
        self.model_name = model_name
        self.batch_size = batch_size
        self.address = address

    def query(self, n):
        self.check_query_num(n)
        path_list = np.array([x[2] for x in self.data_pool])
        # update the inference results.
        while len(self.data_not_inferred) > 0:
            md5_list, uuid_list, data_list = self.proc_func(self.data_not_inferred)
            probs = self.infer_func(data_list, self.batch_size,
                                    model_name=self.model_name, address=self.address)
            for i in range(len(probs)):
                self.db_manager.update_inference(uuid_list[i], probs[i])
            self.data_not_inferred = self.db_manager.get_rows(inferred=False)
        # query the inference table.
        rows = self.db_manager.get_rows()
        infer_probs = [x[3] for x in rows]
        probs_sorted = -np.sort(-np.array(infer_probs))
        difference_list = probs_sorted[:, 0] - probs_sorted[:, 1]
        return path_list[difference_list.argsort()[:n]]


class RatioConfidence(Strategy):
    """
    Ratio of Confidence sampling using Triton Inference Server.
    Reference: Active learning literature survey. (https://minds.wisconsin.edu/handle/1793/60660)
    """

    def __init__(self, infer_func, proc_func, model_name, batch_size, address='localhost:8900'):
        super(RatioConfidence, self).__init__(infer_func=infer_func, proc_func=proc_func)
        self.model_name = model_name
        self.batch_size = batch_size
        self.address = address

    def query(self, n):
        self.check_query_num(n)
        path_list = np.array([x[2] for x in self.data_pool])
        # update the inference results.
        while len(self.data_not_inferred) > 0:
            md5_list, uuid_list, data_list = self.proc_func(self.data_not_inferred)
            probs = self.infer_func(data_list, self.batch_size,
                                    model_name=self.model_name, address=self.address)
            for i in range(len(probs)):
                self.db_manager.update_inference(uuid_list[i], probs[i])
            self.data_not_inferred = self.db_manager.get_rows(inferred=False)
        # query the inference table.
        rows = self.db_manager.get_rows()
        infer_probs = [x[3] for x in rows]
        probs_sorted = -np.sort(-np.array(infer_probs))
        difference_list = probs_sorted[:, 0] / probs_sorted[:, 1]
        return path_list[difference_list.argsort()[:n]]


class EntropySampling(Strategy):
    """
    Entropy Sampling using Triton Inference Server.
    Reference: Active learning literature survey. (https://minds.wisconsin.edu/handle/1793/60660)
    """

    def __init__(self, infer_func, proc_func, model_name, batch_size, address='localhost:8900'):
        super(EntropySampling, self).__init__(infer_func=infer_func, proc_func=proc_func)
        self.model_name = model_name
        self.batch_size = batch_size
        self.address = address

    def query(self, n):
        self.check_query_num(n)
        path_list = np.array([x[2] for x in self.data_pool])
        # update the inference results.
        while len(self.data_not_inferred) > 0:
            md5_list, uuid_list, data_list = self.proc_func(self.data_not_inferred)
            probs = self.infer_func(data_list, self.batch_size,
                                    model_name=self.model_name, address=self.address)
            for i in range(len(probs)):
                self.db_manager.update_inference(uuid_list[i], probs[i])
            self.data_not_inferred = self.db_manager.get_rows(inferred=False)
        # query the inference table.
        rows = self.db_manager.get_rows()
        infer_probs = [x[3] for x in rows]
        entropy = (infer_probs * np.log(infer_probs)).sum(1)
        return path_list[entropy.argsort()[:n]]
