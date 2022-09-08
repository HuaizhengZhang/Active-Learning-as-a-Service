"""
Client example for image classification active learning tasks.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 01/05/2022
"""
import time
from alaas.client import Client


def list_cifar10(file_name, budget=1000):
    client = Client('grpc://0.0.0.0:60035')

    with open(file_name) as file:
        lines = file.readlines()
        remote_file_list = [line.rstrip() for line in lines]

        start_time = time.time()
        queries = client.query_by_uri(remote_file_list, budget=budget)
        end_al_time = time.time()

        with open('log.txt', 'w') as log:
            for q in queries:
                print(q, file=log)

        print(f"Latency: {end_al_time - start_time}")


def part_cifar10(budget=3):
    client = Client('grpc://0.0.0.0:60035')
    remote_file_list = [
        'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png',
        'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane2.png',
        'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane3.png',
        'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane4.png',
        'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane5.png',
        'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane6.png'
    ]
    start_time = time.time()
    queries = client.query_by_uri(remote_file_list, budget=budget)
    end_al_time = time.time()
    print(queries)
    print(f"Latency: {end_al_time - start_time}")


if __name__ == '__main__':
    part_cifar10()
