"""
Client example for image classification active learning tasks.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 01/05/2022
"""
import time
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
        start_time = time.time()
        # push the data urls.
        print(client.push(data_list=url_list, asynchronous=False).text)
        end_download_time = time.time()
        # start querying.
        print(client.query(budget).text)
        end_al_time = time.time()
        return end_al_time - start_time, end_download_time - start_time, end_al_time - end_download_time


if __name__ == '__main__':
    overall_latency, download_latency, al_latency = start_client()
    print(f"overall latency: {overall_latency}\ndownload latency: {download_latency}\nAL latency: {al_latency}")
