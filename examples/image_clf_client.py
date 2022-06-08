"""
Client example for image classification active learning tasks.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 01/05/2022
"""

from alaas.client import Client


def start_client(budget=5):
    remote_file_list = "test_images.txt"
    # prepare the unlabeled data urls.
    with open(remote_file_list) as file:
        url_list = [line.rstrip() for line in file.readlines()]

        # define the ALaaS client, and push data to the server.
        client = Client("http://0.0.0.0:8001")
        # update the server configuration file.
        client.update_config("./resnet_triton.yml")
        # push the data urls.
        client.push(data_list=url_list)
        # start querying.
        # print(client.query(budget).text)


if __name__ == '__main__':
    start_client()
