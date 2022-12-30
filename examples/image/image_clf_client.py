"""
Client example for image classification active learning tasks.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 01/05/2022
"""
import time
from alaas.client import Client


if __name__ == '__main__':
    client = Client('http://0.0.0.0:8081')
    remote_file_list = [
        'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png',
        'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane2.png',
        'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane3.png',
        'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane4.png',
        'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane5.png',
        'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane6.png'
    ]
    start_time = time.time()
    queries = client.query_by_uri(remote_file_list, budget=3)
    end_al_time = time.time()
    print(queries)
    print(f"Latency: {end_al_time - start_time}")
