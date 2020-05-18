FROM ubuntu:18.04
LABEL version="1.1"
RUN apt-get update && apt-get install -y wget && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    apt remove -y python 3.6 python3.5 python2.7 python-minimal && apt autoremove -y && apt-get clean -y && \
    cp /opt/conda/bin/python /usr/bin/python
ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /opt/conda/lib:$LD_LIBRARY_PATH
ADD requirements.txt requirements.txt
ADD project project
RUN pip install -r requirements.txt