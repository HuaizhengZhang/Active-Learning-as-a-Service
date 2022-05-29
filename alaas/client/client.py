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


if __name__ == '__main__':
    server_config_pth = "../../examples/resnet_triton.yml"
    remote_file_list = "../../examples/test_images.txt"
    with open(remote_file_list) as file:
        url_list = [line.rstrip() for line in file.readlines()]

        # define the ALaaS client, and push data to the server.
        client = Client("http://0.0.0.0:8001")
        print(client.push(data_list=url_list).text)
        print(client.query(5).text)
        # print(client.update_config(server_config_pth).json())
