"""
Client example for image classification active learning tasks.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 01/05/2022
"""
import time
from alaas.client import Client

if __name__ == '__main__':
    client = Client('http://0.0.0.0:60035')
    remote_file_list = "test_images.txt"
    # prepare the unlabeled data urls.
    with open(remote_file_list) as file:
        url_list = [line.rstrip() for line in file.readlines()]

        start_time = time.time()
        queries = client.query_by_uri(url_list, budget=10)
        end_al_time = time.time()
        print(queries)
        print(f"Latency: {end_al_time - start_time}")
