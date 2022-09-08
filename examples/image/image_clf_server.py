"""
Server example for image classification active learning tasks.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 01/05/2022
"""

from alaas.server import Server

SERVER_CONFIG = './resnet18.yml'

if __name__ == '__main__':
    # start the server by an input configuration file.
    # Server.start_by_config(SERVER_CONFIG)

    # start the server by parameters. 
    Server.start()
