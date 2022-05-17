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

    def query(self, budget):
        """
        start the active learning process on current data pool with the given budget.
        """
        return requests.get(self.server_url + "/query", params={'budget': budget})


if __name__ == '__main__':
    remote_file_list = "../../examples/test_images.txt"
    with open(remote_file_list) as file:
        url_list = [line.rstrip() for line in file.readlines()]
        print(url_list[:5])

        # define the ALaaS client, and push data to the server.
        client = Client("http://0.0.0.0:8001")
        print(client.push(data_list=url_list[:5]).text)
        print(client.query(10).text)
