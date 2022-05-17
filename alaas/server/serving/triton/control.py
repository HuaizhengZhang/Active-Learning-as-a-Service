#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: May 17, 2022
"""
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: Apr 30, 2022
"""
import docker
from docker.errors import ImageNotFound
from docker.types import Mount, DeviceRequest


def triton_container_run(
        source='nvcr.io',
        triton_version=...,
        triton_variation='py3',
        device: str = 'cpu',
        model_repository_path: str = ...,
        docker_kwargs: dict = None,
        tritonserver_kwargs: dict = None,
):
    """Start Triton server with specific image. If the specified image is not found locally, the
    function will try to pull the image first.

    Args:
        source (str): Source of the Docker Hub. Default to `nvcr.io`.
        triton_version (str): Triton server version in the form of <xx.yy>, e.g., 22.03
        triton_variation (str): Several docker images are available for each version xx.yy
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
        device (str):
        model_repository_path (str): Local path to the Triton model repository. This directory will be
            mounted to the target docker container using the bind type and read only.
        docker_kwargs (dict): Additionally Docker keyword arguments to run the docker container. The
            values provided will override the function pre-defined keyword arguments supplied to the Docker
            container: i.e.,
                detach=True, auto_remove=True, shm_size=1G,
                ports={'8000' 8000, '8001': 8001, '8002': 8002},
                mount=<defined `model_repository_path`> -> `/models/`
        tritonserver_kwargs (dict): Additionally tritonserver start command in key-value pair. The values
            provided will override the function pre-defined command for running the tritonserver: i.e.,
                --model-repository=models
    """
    client = docker.from_env()
    _empty_dict = dict()

    # Get image name
    repo_name = f'{source}/nvidia/tritonserver'
    tag = f'{triton_version}-{triton_variation}'
    image_name = f'{repo_name}:{tag}'

    # Docker keyword arguments
    mounts = [
        Mount(target=f'/models/', source=model_repository_path, type='bind', read_only=True),
    ]
    kwargs = {
        'detach': True, 'auto_remove': True, 'shm_size': '1G',
        'ports': {'8000': 8000, '8001': 8001, '8002': 8002},
        'mounts': mounts,
    }
    kwargs.update(docker_kwargs or _empty_dict)

    # Triton command keyword arguments
    trtserver_kwargs_ = {'model-repository': '/models'}
    trtserver_kwargs_.update(tritonserver_kwargs or _empty_dict)
    tritonserver_command = ' '.join(map(lambda kv: f'--{kv[0]}={kv[1]}', trtserver_kwargs_.items()))

    if device != 'cpu':
        name, device_ids = device.split(':')
        assert name == 'cuda', f'Unrecognized device name: "{name}" in device "{device}"'
        device_ids = device_ids.split(',')
        dr = DeviceRequest(driver='nvidia', device_ids=device_ids, capabilities=[['gpu']])
        kwargs['device_requests'] = [dr]
    try:
        container = client.containers.run(
            image_name, f'tritonserver {tritonserver_command}',
            **kwargs
        )
    except ImageNotFound:
        print(f'Error: image {image_name} not found')
        print(f'Try pulling image {image_name}...')
        client.images.pull(repo_name, tag=tag)
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
