"""
Client example for text classification active learning tasks.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 22/07/2022
"""
import time
from alaas.client import Client

if __name__ == '__main__':
    client = Client('http://0.0.0.0:8081')
    text_list = [
        'The movie itself was to me a huge disappointment.',
        'My name is Wolfgang and I live in Berlin',
        'My name is Wolfgang and I live in Berlin',
        'My name is Wolfgang and I live in Berlin',
        'My name is Wolfgang and I live in Berlin'
    ]
    start_time = time.time()
    queries = client.query_by_text(text_list, budget=3)
    end_al_time = time.time()
    print(queries)
    print(f"Latency: {end_al_time - start_time}")
