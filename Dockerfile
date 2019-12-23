FROM tensorflow/tensorflow:latest-py3
#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
#FROM ben/tensorflow-serving-devel-gpu:latest
RUN apt-get update && \
    apt-get install -y graphviz swig libsm6 libxext6 libxrender-dev libz-dev && \
    pip install wheel tqdm pydot graphviz matplotlib gym[atari] gym[box2d] box2d-py pytest \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app
CMD ["python", "main.py"]
#ENTRYPOINT python
# TODO get mounting for project files working
#RUN pip install -r /app/requirements.txt
