#########################################################
## Python Environment with CUDA
#########################################################

FROM nvidia/cuda:11.4.0-devel-ubuntu20.04 AS python_base_cuda
LABEL MAINTAINER="Anomalib Development Team"

# Setup Proxies
ENV http_proxy=http://proxy-dmz.intel.com:912
ENV https_proxy=http://proxy-dmz.intel.com:912
ENV ftp_proxy=http://proxy-dmz.intel.com:912

# Update system and install wget
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y wget ffmpeg libpython3.8 git sudo

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh --quiet && \
    bash ~/miniconda.sh -b -p /opt/conda
ENV PATH "/opt/conda/bin:${PATH}" && conda install python=3.8


#########################################################
## Anomalib Development Env
#########################################################

FROM python_base_cuda as anomalib_development_env

# Get MVTec Dataset
# cache datasets first as changes to requirements do no affect this stage
RUN wget ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz --quiet && \
    mkdir -p /tmp/anomalib/datasets/MVTec && \
    tar -xf mvtec_anomaly_detection.tar.xz -C /tmp/anomalib/datasets/MVTec

# Install all anomalib requirements
COPY ./requirements/base.txt /tmp/anomalib/requirements/base.txt
RUN pip install -r /tmp/anomalib/requirements/base.txt

COPY ./requirements/openvino.txt /tmp/anomalib/requirements/openvino.txt
RUN pip install -r /tmp/anomalib/requirements/openvino.txt

# Install other requirements related to development
COPY ./requirements/dev.txt /tmp/anomalib/requirements/dev.txt
RUN pip install -r /tmp/anomalib/requirements/dev.txt
