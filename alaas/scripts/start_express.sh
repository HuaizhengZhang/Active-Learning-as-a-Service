#!/usr/bin/env bash
docker run --name mongo_express \
       --network mongo_network \
       -p 8081:8081 \
       -e ME_CONFIG_MONGODB_SERVER=mongo \
       -e ME_CONFIG_MONGODB_ADMINUSERNAME=yizheng \
       -e ME_CONFIG_MONGODB_ADMINPASSWORD=yizheng \
       -e ME_CONFIG_BASICAUTH_USERNAME=yizheng \
       -e ME_CONFIG_BASICAUTH_PASSWORD=yizheng \
       -d mongo-express:latest