"""
Server example for image classification active learning tasks.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 01/05/2022
"""

from alaas.server import Server

SERVER_CONFIG = './resnet_triton.yml'

if __name__ == '__main__':
    Server(config_path=SERVER_CONFIG).start(host="0.0.0.0", port=8001)
