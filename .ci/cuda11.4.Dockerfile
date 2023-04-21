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
        curl \
        wget \
        ffmpeg \
        libpython3.8 \
        npm \
        pandoc \
        ruby \
        software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install latest git for github actions
RUN add-apt-repository ppa:git-core/ppa &&\
    apt-get update && \
    apt-get install --no-install-recommends -y git &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Prettier requires atleast nodejs 10 and actions/checkout requires nodejs 16
RUN curl -sL https://deb.nodesource.com/setup_current.x > nodesetup.sh && \
    bash - nodesetup.sh && \
    apt-get install --no-install-recommends -y nodejs && \
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
