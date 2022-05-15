#!/usr/bin/env bash
docker run -it \
       -p 27017:27017 --name mongo \
       --network mongo-network \
       -e MONGO_INITDB_ROOT_USERNAME=yizheng \
       -e MONGO_INITDB_ROOT_PASSWORD=yizheng \
       -d mongo:latest