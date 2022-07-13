import requests


class Client:
    """
    Client: the active learning client class.
    """

    def __init__(self, server_url: str):
        self.server_url = server_url

    def push(self, data_list, asynchronous=False):
        """
        push the data url list to the cloud active learning server.
        """
        return requests.post(self.server_url + "/push", params={'asynchronous': asynchronous}, json=data_list)

    def update_config(self, new_config):
        """
        update the server configuration.
        """
        return requests.post(self.server_url + "/update_cfg", files={'config': open(new_config, 'rb')})

    def query(self, budget):
        """
        start the active learning process on current data pool with the given budget.
        """
        return requests.get(self.server_url + "/query", params={'budget': budget})
