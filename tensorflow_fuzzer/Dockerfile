FROM tensorflow/tensorflow:2.7.0-gpu
# FROM tensorflow/tensorflow:2.1.0-gpu-p3
RUN pip3 install scipy bs4
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt-get install -y sqlite3 gringo curl
COPY original_tensorflow/tensorflow/compiler/tests/   /usr/local/lib/python3.8/dist-packages/tensorflow/compiler/tests

WORKDIR /workspace/tensorflow_skipfuzz/tensorflow