FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS dev

RUN apt-get update && apt-get install -y git unzip wget vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip3 install learning3d 
RUN pip3 install -r requirements.txt
 
RUN wget https://huggingface.co/datasets/Msun/modelnet40/resolve/d5dc795541800feeb7a4b3bd3142729a0d2adf7a/modelnet40_ply_hdf5_2048.zip?download=true -O model.zip && \
    mkdir /ai && \
    unzip model.zip -d /ai/data

FROM dev AS run

COPY . /app
WORKDIR /app

CMD ["python", "/app/src/app.py", "--pretrained_c", "/app/pretrained/best_model_cls.t7"]