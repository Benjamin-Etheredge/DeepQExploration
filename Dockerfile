FROM tensorflow/tensorflow:latest-gpu
COPY requirements.txt /app/
WORKDIR /app
RUN apt-get update && \
    apt-get install -y git graphviz swig libsm6 libxext6 libxrender-dev libz-dev && \
    pip install --upgrade pip \
    pip install -r requirements.txt \ 
    pip install mlflow boto3 \
    && rm -rf /var/lib/apt/lists/
COPY ./src /app/src
COPY ./test /app/test
