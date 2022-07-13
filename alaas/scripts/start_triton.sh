#!/usr/bin/env bash
docker run --rm -p8900:8000 -p8901:8001 -p8902:8002 \
  --shm-size=1g --ulimit memlock=-1 \
  -v${PWD}/../example/pytorch/resnet:/models/resnet/ \
  -v${PWD}/../example/pytorch/my-pytorch.tar.gz:/models/my-pytorch.tar.gz \
  nvcr.io/nvidia/tritonserver:22.04-pyt-python-py3 tritonserver --model-repository=/models
