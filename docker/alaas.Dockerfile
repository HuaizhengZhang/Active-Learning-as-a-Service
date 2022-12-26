# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster


WORKDIR /server

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install alaas boto3
RUN pip3 install torch torchvision

COPY ./entry.py .

CMD [ "python3", "./entry.py", "--config", "./config.yml"]