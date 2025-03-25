FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update && apt-get install -y git unzip wget vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install learning3d