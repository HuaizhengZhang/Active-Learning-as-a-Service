"""
Base active learning class.
@author huangyz0918 (huangyz0918@gmail.com)
@date 20/07/2022
"""


class Strategy:
    """The basic active learning strategy class"""

    def __init__(self, pool_size, path_mapping):
        """
        @param pool_size: if pool size = 1, the settings is stream-based.
        @param path_mapping: the mapping with input uris.
        """
        self.pool_size = pool_size
        self.path_mapping = path_mapping

    def check_query_num(self, number):
        """
        Check the input query number is valid or not.
        """
        if number > self.pool_size:
            raise ValueError(f"please indicate a query number smaller than data pool size: {self.pool_size}")

    def query(self, number, embeddings):
        """
        The query function that calls the specific active learning strategy.
        :param number: the query number.
        :@param embeddings: the data embedding.
        :return: the data ids in a numpy array.
        """
        pass
