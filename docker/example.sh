#!/bin/bash

# An example of starting ALaaS server using docker.
docker run -it --rm -p 8081:8081 \
        --mount type=bind,source=/Users/huangyz0918/Desktop/Active-Learning-as-a-Service/examples/image/resnet18.yml,target=/server/config.yml,readonly huangyz0918/alaas:v1