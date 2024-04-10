FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

WORKDIR /opt/app
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
COPY docker/requirements.txt /opt/app
RUN python -m pip install -r requirements.txt
#RUN python -m pip install fever-scorer==2.0.39

ENV PYTHONIOENCODING=utf-8
WORKDIR /workspace
ENV DIR_TMP /workspace

