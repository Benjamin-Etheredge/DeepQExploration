FROM tensorflow/tensorflow:latest-py3
RUN apt-get update && \
    apt-get install -y graphviz && \
    pip install tqdm pydot graphviz matplotlib gym
RUN apt-get install -y swig
RUN pip install gym[box2d] Box2D
COPY . /app
CMD ["python", "/app/main.py"]
# TODO get mounting for project files working
#RUN pip install -r /app/requirements.txt
#CMD python /app/main.py
