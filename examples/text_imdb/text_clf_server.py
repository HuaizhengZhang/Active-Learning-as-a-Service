"""
Server example for text classification active learning tasks.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 23/07/2022
"""

from alaas.server import Server

SERVER_CONFIG = './bert_imdb.yml'

if __name__ == '__main__':
    Server(SERVER_CONFIG).start()
