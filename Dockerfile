# no GPU
# FROM mcr.microsoft.com/vscode/devcontainers/base:0-ubuntu-20.04

# with GPU
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python3.10 python3-pip

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt
