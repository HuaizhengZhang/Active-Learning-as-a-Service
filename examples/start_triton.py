#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: May 19, 2022
"""
from alaas.server.serving.triton.control import triton_container_run, triton_container_stop
from alaas.server.serving.triton.converter import TritonPythonModelConverter
from alaas.server.serving.triton.env_exporter import CondaEnvExporter
from alaas.server.util import ConfigManager

if __name__ == '__main__':
    config_path = 'resnet_triton.yml'
    cfg_mgr = ConfigManager(config_path)

    # prepare init model
    infer_model_config = cfg_mgr.strategy.infer_model
    worker_config = cfg_mgr.al_server.worker
    model_repository_path = worker_config.model_repository_path

    # convert the requested infer model to Triton Python model, and save to local model repository
    env_exporter = CondaEnvExporter(model_repository_path=model_repository_path)
    env_exporter.export(infer_model_config.conda_env)
    converter = TritonPythonModelConverter(model_repository_path=model_repository_path)
    converter.from_torch_hub(infer_model_config)

    # serve the infer model with Triton
    triton_container = triton_container_run(**worker_config.dict())

    # do some active learning...
    ...

    # stop the infer model
    triton_container_stop(triton_container.id)
