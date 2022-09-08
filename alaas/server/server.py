"""
The basic server of ALaaS.
@author huangyz0918 (huangyz0918@gmail.com)
@date 20/07/2022
"""
from jina import Flow

from alaas.server.util import ConfigManager
from alaas.server.executors import TorchALWorker


class Server:
    """
    Server: Server Class for Active Learning Services.
    """
    @staticmethod
    def start(proto='http',
                port=8081, 
                host='0.0.0.0', 
                job_name='default_app', 
                model_hub='pytorch/vision:v0.10.0', 
                model_name='resnet18', 
                device='cpu', 
                strategy='LeastConfidence', 
                batch_size=1, 
                replica=1, 
                tokenizer=None, 
                transformers_task=None
            ):
        # start with an image classification application (with PyTorch resnet18) as the example.
        Flow(protocol=proto, port=port, host=host) \
            .add(name=job_name,
                 uses=TorchALWorker,
                 uses_with={
                     'model_name': model_name,
                     'model_repo': model_hub,
                     'device': device,
                     'strategy': strategy,
                     'minibatch_size': batch_size,
                     'tokenizer_model': tokenizer,
                     'task': transformers_task
                 },
                 replicas=replica) \
            .start()


    @staticmethod
    def start_by_config(config_path):
        # TODO: multi-worker support
        cfg_manager = ConfigManager(config_path)
        _host = cfg_manager.al_worker.host
        _port = cfg_manager.al_worker.port
        _replica = cfg_manager.al_worker.replicas
        _name = cfg_manager.name
        _proto = cfg_manager.al_worker.protocol

        # Active learning executor parameters.
        _strategy = cfg_manager.strategy.type.value
        _model_hub = cfg_manager.strategy.model.hub
        _model_name = cfg_manager.strategy.model.name
        _device = cfg_manager.strategy.model.device
        _batch_size = cfg_manager.strategy.model.batch_size
        # only for text data/model.
        _tokenizer = cfg_manager.strategy.model.tokenizer
        _task = cfg_manager.strategy.model.task

        Flow(protocol=_proto, port=_port, host=_host) \
            .add(name=_name,
                 uses=TorchALWorker,
                 uses_with={
                     'model_name': _model_name,
                     'model_repo': _model_hub,
                     'device': _device,
                     'strategy': _strategy,
                     'minibatch_size': _batch_size,
                     'tokenizer_model': _tokenizer,
                     'task': _task
                 },
                 replicas=_replica) \
            .start()
