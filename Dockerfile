FROM tensorflow/tensorflow:latest-gpu-py3
#FROM python:3.6.
#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.0
#FROM ben/tensorflow-serving-devel-gpu:lates
RUN apt-get update && 
    apt-get install -y graphviz swig libsm6 libxext6 libxrender-dev libz-dev && 
    pip install objgraph memory_profiler guppy3 wheel pydot graphviz matplotlib gym[atari] gym[box2d] box2d-py numba 
    && rm -rf /var/lib/apt/lists/
#COPY tensorflow-2.1.0-cp36-cp36m-linux_x86_64.whl tensorflow-2.1.0-cp36-cp36m-linux_x86_64.wh
#RUN pip ninstall tensorflow && pip install tensorflow-2.1.0-cp36-cp36m-linux_x86_64.whl
COPY . /app
WORKDIR /app
#ENTRYPOINT python
CMD bash
# TODO get mounting for project files working
#RUN pip install -r /app/requirements.txt
