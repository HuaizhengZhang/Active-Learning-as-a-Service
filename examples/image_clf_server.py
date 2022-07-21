"""
Server example for image classification active learning tasks.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 01/05/2022
"""

from alaas.server import Server

SERVER_CONFIG = './resnet_cifar.yml'

if __name__ == '__main__':
    Server(SERVER_CONFIG).start()
