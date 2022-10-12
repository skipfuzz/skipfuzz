FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

#FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
# FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
#FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN pip3 install scipy bs4 expecttest hypothesis
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt-get install -y sqlite3 gringo curl


