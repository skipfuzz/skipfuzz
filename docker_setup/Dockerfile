FROM tensorflow/tensorflow:2.8.0-gpu
RUN pip3 install scipy bs4 coverage
RUN apt-get update
RUN apt-get install sqlite3
COPY original_tensorflow/tensorflow/compiler/tests/   /usr/local/lib/python3.8/dist-packages/tensorflow/compiler/tests
