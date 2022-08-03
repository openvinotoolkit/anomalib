#########################################################
## Python Environment with CUDA
#########################################################

FROM nvidia/cuda:11.4.0-devel-ubuntu20.04 AS python_base_cuda
LABEL MAINTAINER="Anomalib Development Team"

# Update system and install wget
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
        wget=1.20.3-1ubuntu2 \
        ffmpeg=7:4.2.7-0ubuntu0.1 \
        libpython3.8=3.8.10-0ubuntu1~20.04.5 \
        git=1:2.25.1-1ubuntu3.5 \
        sudo=1.8.31-1ubuntu1.2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh --quiet && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH "/opt/conda/bin:${PATH}"
RUN conda install python=3.8


#########################################################
## Anomalib Development Env
#########################################################

FROM python_base_cuda as anomalib_development_env

# Install all anomalib requirements
COPY ./requirements/base.txt /tmp/anomalib/requirements/base.txt
RUN pip install --no-cache-dir -r /tmp/anomalib/requirements/base.txt

COPY ./requirements/openvino.txt /tmp/anomalib/requirements/openvino.txt
RUN pip install --no-cache-dir -r /tmp/anomalib/requirements/openvino.txt

# Install other requirements related to development
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
        nodejs=10.19.0~dfsg-3ubuntu1 \
        npm=6.14.4+ds-1ubuntu2 \
        ruby=1:2.7+1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY ./requirements/dev.txt /tmp/anomalib/requirements/dev.txt
RUN pip install --no-cache-dir -r /tmp/anomalib/requirements/dev.txt

# Install anomalib
COPY . /anomalib
WORKDIR /anomalib
RUN pip install --no-cache-dir -e .
