# This image was needed to use python on remote docker host
FROM tensorflow/tensorflow:latest-gpu-py3
#FROM ben/tensorflow-serving-devel-gpu:latest
ARG ssh_pub_key

RUN apt-get update &&\
    apt-get install -y graphviz swig libsm6 libxext6 libxrender-dev libz-dev &&\
    apt-get install -y openssh-server vim &&\
    pip install guppy3 wheel tqdm pydot graphviz matplotlib gym[atari] gym[box2d] box2d-py pytest &&\
    rm -rf /var/lib/apt/lists/*

    # Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh && \
    echo "$ssh_pub_key" > /root/.ssh/authorized_keys && \
    mkdir /var/run/sshd && \
    sed 's@.*Subsystem.*sftp.*@Subsystem sftp internal-sftp@g' -i /etc/ssh/sshd_config
    #sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
#COPY /home/ben/.ssh /root/.ssh
#ENV NOTVISIBLE "in users profile" && \
#RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
#CMD ["bash"]
#CMD /usr/sbin/sshd && /bin/bash