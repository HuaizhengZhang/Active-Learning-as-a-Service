"""
Client example for text classification active learning tasks.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 22/07/2022
"""
import time
from alaas.client import Client

if __name__ == '__main__':
    client = Client('grpc://0.0.0.0:60035')
    text_list = [
        'The movie itself was to me a huge disappointment.',
        'What a nasty cynical film.',
        'I didn\'t know this came from Canada, but it is very good. Very good!',
        'I\'m a male, not given to women\'s movies, but this is really a well done special story.',
        'I admit, the great majority of films released before say 1933 are just not for me.'
    ]
    start_time = time.time()
    queries = client.query_by_text(text_list, budget=3)
    end_al_time = time.time()
    print(queries)
    print(f"Latency: {end_al_time - start_time}")
