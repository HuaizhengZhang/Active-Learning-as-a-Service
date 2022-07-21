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
        # TODO: multi-worker support
        self.cfg_manager = ConfigManager(config_path)
        self._host = self.cfg_manager.al_worker.host
        self._port = self.cfg_manager.al_worker.port
        self._replica = self.cfg_manager.al_worker.replicas
        self._name = self.cfg_manager.name
        self._proto = self.cfg_manager.al_worker.protocol

        # Active learning executor parameters.
        self._strategy = self.cfg_manager.strategy.type.value
        self._executor_name = self.cfg_manager.strategy.model.name
        self._model_hub = self.cfg_manager.strategy.model.hub
        self._model_name = self.cfg_manager.strategy.model.model
        self._device = self.cfg_manager.strategy.model.device
        self._batch_size = self.cfg_manager.strategy.model.batch_size

    def start(self):
        Flow(protocol=self._proto, port=self._port, host=self._host) \
            .add(name=self._name,
                 uses=TorchWorker,
                 uses_with={
                     'model_name': self._model_name,
                     'model_repo': self._model_hub,
                     'device': self._device,
                     'strategy': self._strategy,
                     'minibatch_size': self._batch_size,
                 },
                 replicas=self._replica) \
            .start()
