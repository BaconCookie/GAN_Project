FROM nvidia/cuda:9.0-cudnn7-devel
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 
ENV PATH /opt/conda/bin:$PATH 

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates git mercurial subversion bash apt-utils python3

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda 
RUN rm ~/anaconda.sh
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate base" >> ~/.bashrc

RUN conda install python=3.6

RUN conda install pip

RUN pip install --upgrade tensorflow-gpu
RUN pip install keras
RUN conda install -c conda-forge matplotlib
RUN conda install -c conda-forge seaborn

RUN pip install Pillow

RUN apt-get install -y curl grep sed dpkg && \
 TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
 curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
 dpkg -i tini.deb && \
 rm tini.deb && \
 apt-get clean

RUN mkdir /mounted-files
RUN cd /mounted-files

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--notebook-dir=/mounted-files" ]
