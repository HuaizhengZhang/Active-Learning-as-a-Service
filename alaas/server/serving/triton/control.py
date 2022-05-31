#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: May 17, 2022
"""
import os
from enum import Enum
from typing import Union, Optional, List

import docker
from docker.errors import ImageNotFound
from docker.types import Mount, DeviceRequest


def triton_container_run(
        docker_repo='nvcr.io/nvidia/tritonserver',
        version: str = ...,
        variation: Union[str, Enum] = 'py3',
        gpus: Optional[Union[List[Union[int, str]], int, str]] = ...,
        model_repository_path: os.PathLike = ...,
        docker_kwargs: dict = None,
        command: dict = None,
        **kwargs,  # noqa
):
    """Start Triton server with specific image. If the specified image is not found locally, the
    function will try to pull the image first.

    Args:
        docker_repo (str): Source of the Docker repository. Default to `nvcr.io/nvidia/tritonserver`.
        version (str): Triton server version in the form of <xx.yy>, e.g., 22.03
        variation (Union[str, Enum]): Several docker images are available for each version xx.yy
            (refer from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver):
            - The `py3` image contains the Triton inference server with support for
              Tensorflow, PyTorch, TensorRT, ONNX and OpenVINO models.
            - The `py3-sdk` image contains Python and C++ client libraries, client
              examples, and the Model Analyzer.
            - The `py3-min` image is used as the base for creating custom Triton server
              containers as described in Customize Triton Container.
            - The `pyt-python-py3` image contains the Triton Inference Server with
              support for PyTorch and Python backends only.
            - The `tf2-python-py3` image contains the Triton Inference Server with
              support for TensorFlow 2.x and Python backends only.
        gpus: Set GPU devices passed to Docker container. Possible values examples:
            - [0, 1, 2] or ['GPU-fef8089b']: a list of GPU UUID(s) or index(es).
            - 1 or 2: number of devices will be used.
            - 'all' or -1: all GPUs will be accessible, this is the default value in base CUDA container images.
            - None: no GPU will be accessible, but driver capabilities will be enabled.
            - unset (Ellipsis): `nvidia-container-runtime` will have the same behavior as `runc`
                (i.e., neither GPUs nor capabilities are exposed).
            Check Nvidia-Docker documentation for reference:
              https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration
        model_repository_path (os.PathLike): Local path to the Triton model repository. This directory will be
            mounted to the target docker container using the bind type and read only.
        docker_kwargs (dict): Additionally Docker keyword arguments to run the docker container. The
            values provided will override the function pre-defined keyword arguments supplied to the Docker
            container: i.e.,
                detach=True, auto_remove=True, shm_size=1G,
                ports={'8000' 8000, '8001': 8001, '8002': 8002},
                mount=<defined `model_repository_path`> -> `/models/`
        command (dict): Additionally tritonserver start command in key-value pair. The values
            provided will override the function pre-defined command for running the tritonserver: i.e.,
                --model-repository=models
    """
    client = docker.from_env()
    _empty_dict = dict()

    # Get image name
    if isinstance(variation, Enum):
        variation = variation.value
    tag = f'{version}-{variation}'
    image_name = f'{docker_repo}:{tag}'

    # Docker keyword arguments
    mounts = [
        Mount(target=f'/models/', source=str(model_repository_path), type='bind', read_only=True),
    ]
    kwargs = {
        'detach': True, 'auto_remove': True, 'shm_size': '1G',
        'ports': {'8000': 8000, '8001': 8001, '8002': 8002},
        'mounts': mounts, 'device_requests': list(),
    }
    kwargs.update(docker_kwargs or _empty_dict)

    # Triton command keyword arguments
    trtserver_kwargs_ = {'model-repository': '/models'}
    trtserver_kwargs_.update(command or _empty_dict)
    tritonserver_command = ' '.join(map(lambda kv: f'--{kv[0]}={kv[1]}', trtserver_kwargs_.items()))

    if gpus is None:
        gpus = 0
    elif gpus == 'all':
        gpus = -1

    if gpus is ...:
        # Neither GPUs nor capabilities are exposed
        pass
    elif isinstance(gpus, int) or gpus == 'all':
        dr = DeviceRequest(driver='nvidia', count=gpus, capabilities=[['gpu']])
        kwargs['device_requests'].append(dr)
    elif isinstance(gpus, list):
        dr = DeviceRequest(driver='nvidia', device_ids=gpus, capabilities=[['gpu']])
        kwargs['device_requests'].append(dr)

    try:
        container = client.containers.run(
            image_name, f'tritonserver {tritonserver_command}',
            **kwargs
        )
    except ImageNotFound:
        print(f'Error: image {image_name} not found')
        print(f'Try pulling image {image_name}...')
        client.images.pull(docker_repo, tag=tag)
        container = client.containers.run(
            image_name, f'tritonserver {tritonserver_command}',
            **kwargs
        )

    return container


def triton_container_stop(container_id):
    client = docker.from_env()
    container = client.containers.get(container_id=container_id)
    container.stop()


def model_load_update(url, config):
    pass


def model_unload_all(url):
    pass
