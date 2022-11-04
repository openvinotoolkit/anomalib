#########################################################
## Python Environment with CUDA
#########################################################
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

FROM nvidia/cuda:11.4.0-devel-ubuntu20.04 AS python_base_cuda11.4
LABEL maintainer="Anomalib Development Team"

# Setup proxies

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV DEBIAN_FRONTEND="noninteractive"

# Update system and install wget
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        curl=7.68.0-1ubuntu2.13 \
        wget=1.20.3-1ubuntu2 \
        ffmpeg=7:4.2.7-0ubuntu0.1 \
        libpython3.8=3.8.10-0ubuntu1~20.04.5 \
        nodejs=10.19.0~dfsg-3ubuntu1 \
        npm=6.14.4+ds-1ubuntu2 \
        pandoc=2.5-3build2 \
        ruby=1:2.7+1 \
        software-properties-common=0.99.9.8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install latest git for github actions
RUN add-apt-repository ppa:git-core/ppa &&\
    apt-get update && \
    apt-get install --no-install-recommends -y git=1:2.38.1-0ppa1~ubuntu20.04.1 &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m user
USER user

# Install Conda
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > ~/miniconda.sh -s && \
    bash ~/miniconda.sh -b -p /home/user/conda && \
    rm ~/miniconda.sh
ENV PATH "/home/user/conda/bin:${PATH}"
RUN conda install python=3.8


#########################################################
## Anomalib Development Env
#########################################################

FROM python_base_cuda11.4 as anomalib_development_env

# Install all anomalib requirements
COPY ./requirements/base.txt /tmp/anomalib/requirements/base.txt
RUN pip install --no-cache-dir -r /tmp/anomalib/requirements/base.txt

COPY ./requirements/openvino.txt /tmp/anomalib/requirements/openvino.txt
RUN pip install --no-cache-dir -r /tmp/anomalib/requirements/openvino.txt

# Install other requirements related to development
COPY ./requirements/dev.txt /tmp/anomalib/requirements/dev.txt
RUN pip install --no-cache-dir -r /tmp/anomalib/requirements/dev.txt

WORKDIR /home/user
