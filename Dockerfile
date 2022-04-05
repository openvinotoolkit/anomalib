#########################################################
## Python Environment with CUDA
#########################################################

FROM nvidia/cuda:11.4.0-devel-ubuntu20.04 AS python_base_cuda
LABEL MAINTAINER="Anomalib Development Team"

# Update system and install wget
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y wget ffmpeg libpython3.8 git sudo

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh --quiet && \
    bash ~/miniconda.sh -b -p /opt/conda
ENV PATH "/opt/conda/bin:${PATH}"
RUN conda install python=3.8


#########################################################
## Anomalib Development Env
#########################################################

FROM python_base_cuda as anomalib_development_env

# Install all anomalib requirements
COPY ./requirements/base.txt /tmp/anomalib/requirements/base.txt
RUN pip install -r /tmp/anomalib/requirements/base.txt

COPY ./requirements/openvino.txt /tmp/anomalib/requirements/openvino.txt
RUN pip install -r /tmp/anomalib/requirements/openvino.txt

# Install other requirements related to development
COPY ./requirements/dev.txt /tmp/anomalib/requirements/dev.txt
RUN pip install -r /tmp/anomalib/requirements/dev.txt

# Install anomalib
COPY . /anomalib
WORKDIR /anomalib
RUN pip install -e .
