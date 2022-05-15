class Strategy:
    def __init__(self, infer_func, source_data):
        self.infer_func = infer_func
        self.source_data = source_data

    def query(self, number):
        """
        The query function that calls the specific active learning strategy.
        :param number: the query number.
        :return: the data ids in a numpy array.
        """
        pass
