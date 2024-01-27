FROM mcr.microsoft.com/vscode/devcontainers/base:0-ubuntu-20.04

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python3.10 python3-pip

WORKDIR /app

RUN mkdir logs

RUN git clone --depth 1 https://github.com/pat-coady/pytorch-sandbox.git

WORKDIR /app/pytorch-sandbox

RUN pip install -r requirements.txt
