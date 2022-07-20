"""
The basic server of ALaaS.
@author huangyz0918 (huangyz0918@gmail.com)
@date 20/07/2022
"""
from jina import Flow

from alaas.server.util import ConfigManager
from alaas.server.executors import TorchWorker


class Server:
    """
    Server: Server Class for Active Learning Services.
    """

    def __init__(self, config_path):
        self.cfg_manager = ConfigManager(config_path)

    def start(self, port=65335, replicas=1):
        Flow(port=port).add(name='worker_1', uses=TorchWorker, replicas=replicas).start()
