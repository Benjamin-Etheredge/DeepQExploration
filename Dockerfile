FROM tensorflow/tensorflow:latest-gpu-py3
#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
#FROM ben/tensorflow-serving-devel-gpu:latest
RUN apt-get update && \
    apt-get install -y graphviz swig libsm6 libxext6 libxrender-dev libz-dev && \
    pip install objgraph memory_profiler guppy3 wheel pydot graphviz matplotlib gym[atari] gym[box2d] box2d-py \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app
#ENTRYPOINT python
CMD bash
# TODO get mounting for project files working
#RUN pip install -r /app/requirements.txt
