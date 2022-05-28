from pathlib import Path
from alaas.server.util import DBManager


class Strategy:
    # TODO: refactor the source data using the database records.
    def __init__(self, infer_func, source_data):
        self.infer_func = infer_func
        self.source_data = source_data
        home_path = str(Path.home())
        self.alaas_home = home_path + "/.alaas/"
        self.db_manager = DBManager(self.alaas_home + 'index.db')

    def query(self, number):
        """
        The query function that calls the specific active learning strategy.
        :param number: the query number.
        :return: the data ids in a numpy array.
        """
        pass
