"""
1. Configuration: ./config.yml
2. Inference  Server: Nvidia triton inference server, CPU (AWS)
3. Dataset: CIFAR-10
4. Model: ResNet-50 PyTorch
5. Storage: AWS S3 (ap-southeast-1)
6. Data Pool Size: 50000
7. Budget: 1000
8. Active Learning Strategy: the Least Confidence Sampling
9. Baseline: DeepAL, ModAL (Block)
10. Baseline Workflow: Download Data -> Inference All -> Active Learning.
11. Our Workflow: Download Data + Inference Each (Async) -> Active Learning.
"""

import time

from alaas.server import Server
from alaas.client import Client


def start_server():
    """
    Start the server by configuration.
    """
    SERVER_CONFIG = './config.yml'

    if __name__ == '__main__':
        Server(config_path=SERVER_CONFIG).start(host="0.0.0.0", port=8001)


def start_client(budget=5):
    """
    Start the client.
    """
    # TODO: replace with AWS S3.
    remote_file_list = "test_images.txt"
    # prepare the unlabeled data urls.
    with open(remote_file_list) as file:
        url_list = [line.rstrip() for line in file.readlines()]

        # define the ALaaS client, and push data to the server.
        client = Client("http://0.0.0.0:8001")
        # update the server configuration file.
        client.update_config("./config.yml")
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
