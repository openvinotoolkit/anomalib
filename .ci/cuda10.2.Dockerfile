#########################################################
## Python Environment with CUDA
#########################################################
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

FROM nvidia/cuda:10.2-devel-ubuntu18.04 AS python_base_cuda10.2
LABEL maintainer="Anomalib Development Team"

# Setup proxies

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV DEBIAN_FRONTEND="noninteractive"

# Update system and install wget
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        curl=7.58.0-2ubuntu3.20 \
        wget=1.19.4-1ubuntu2 \
        ffmpeg=7:3.4.2-2 \
        libpython3.8=3.8.0-3ubuntu1~18.04.2 \
        npm=3.5.2-0ubuntu4 \
        pandoc=1.19.2.4~dfsg-1build4 \
        ruby=1:2.5.1 \
        software-properties-common=0.96.24.32.18 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install latest git for github actions
RUN add-apt-repository ppa:git-core/ppa &&\
    apt-get update && \
    apt-get install --no-install-recommends -y git=1:2.38.1-0ppa1~ubuntu18.04.1 &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Prettier requires atleast nodejs 10
RUN curl -sL https://deb.nodesource.com/setup_14.x > nodesetup.sh && \
    bash - nodesetup.sh && \
    apt-get install --no-install-recommends -y nodejs=14.20.1-1nodesource1 && \
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

FROM python_base_cuda10.2 as anomalib_development_env

# Install all anomalib requirements
COPY ./requirements/base.txt /tmp/anomalib/requirements/base.txt
RUN pip install --no-cache-dir -r /tmp/anomalib/requirements/base.txt

COPY ./requirements/openvino.txt /tmp/anomalib/requirements/openvino.txt
RUN pip install --no-cache-dir -r /tmp/anomalib/requirements/openvino.txt

# Install other requirements related to development
COPY ./requirements/dev.txt /tmp/anomalib/requirements/dev.txt
RUN pip install --no-cache-dir -r /tmp/anomalib/requirements/dev.txt

WORKDIR /home/user
