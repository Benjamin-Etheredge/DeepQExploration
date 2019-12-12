FROM tensorflow/tensorflow:latest-py3
RUN apt-get update && \
    apt-get install -y graphviz swig && \
    pip install tqdm pydot graphviz matplotlib gym gym[box2d] Box2D \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR . /app
CMD ["python", "main.py"]
# TODO get mounting for project files working
#RUN pip install -r /app/requirements.txt
#CMD python /app/main.py
