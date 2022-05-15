import numpy as np
from .base import Strategy
from alaas.server.serving import ServeClient
from alaas.server.util import chunks


class LeastConfidence(Strategy):
    """
    Least Confidence Sampling.
    Reference: A Sequential Algorithm for Training Text Classifiers (1994).
    (https://arxiv.org/pdf/cmp-lg/9407020)
    """

    def __init__(self, infer_func, source_data):
        super(LeastConfidence, self).__init__(infer_func, source_data)

    def query(self, n):
        probs = self.infer_func(self.source_data, is_prob=True)
        uncertainties = np.amax(probs, axis=1)
        return uncertainties.argsort()[:n]


class LeastConfidenceTriton(Strategy):
    """
    Least Confidence Sampling using Triton Inference Server.
    Reference: A Sequential Algorithm for Training Text Classifiers (1994).
    (https://arxiv.org/pdf/cmp-lg/9407020)
    """

    def __init__(self, source_data, model_name, batch_size, address='localhost:8900'):
        super(LeastConfidenceTriton, self).__init__(None, source_data)
        self.model_name = model_name
        self.batch_size = batch_size
        self.address = address

    def query(self, n):
        probs = []
        triton_client = ServeClient()
        for batch in chunks(self.source_data, self.batch_size):
            for item in triton_client.infer(batch, self.model_name, address=self.address):
                probs.append(item)
        uncertainties = np.amax(probs, axis=1)
        return uncertainties.argsort()[:n]
