#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: May 19, 2022
"""
import yaml

from alaas.server.serving.triton.control import triton_container_run, triton_container_stop
from alaas.server.util import Config

if __name__ == '__main__':
    config_path = 'resnet_triton_local.yml'
    with open(config_path) as f:
        config_obj = yaml.safe_load(f)
    config = Config.parse_obj(config_path)

    # prepare init model
    infer_model_config = config.strategy.infer_model
    # convert the requested infer model to Triton Python model, and save to local model repository
    converter = TritonPythonModelConverter(model_repository_path=...)
    converter.from_torch_hub(infer_model_config)

    worker_config = config.al_server.worker
    # serve the infer model with Triton
    triton_container = triton_container_run(**worker_config.dict())

    # do some active learning...
    ...

    # stop the infer model
    triton_container_stop(triton_container.id)
