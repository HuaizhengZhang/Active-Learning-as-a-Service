from pathlib import Path
from alaas.server.util import DBManager


class Strategy:
    """The basic active learning strategy class"""

    def __init__(self, infer_func, proc_func):
        self.infer_func = infer_func
        self.proc_func = proc_func
        home_path = str(Path.home())
        self.alaas_home = home_path + "/.alaas/"
        self.db_manager = DBManager(self.alaas_home + 'index.db')
        self.data_pool = self.db_manager.read_records()
        self.data_inferred = self.db_manager.get_rows()
        self.data_not_inferred = self.db_manager.get_rows(inferred=False)

    def check_query_num(self, number):
        """
        Check the input query number is valid or not.
        """
        pool_size = len(self.data_pool)
        if number > pool_size:
            raise ValueError(f"please indicate a query number smaller than data pool size: {pool_size}")

    def query(self, number):
        """
        The query function that calls the specific active learning strategy.
        :param number: the query number.
        :return: the data ids in a numpy array.
        """
        pass
