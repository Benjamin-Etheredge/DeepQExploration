FROM tensorflow/tensorflow:latest-gpu-py3
COPY requirements.txt /app
WORKDIR /app
RUN apt-get update && \
    apt-get install -y graphviz swig libsm6 libxext6 libxrender-dev libz-dev && \
    pip install -r requirements.txt \ 
    && rm -rf /var/lib/apt/lists/
COPY ./src /app
