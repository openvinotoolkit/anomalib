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
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y wget ffmpeg libpython3.8

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /opt/conda
ENV PATH "/opt/conda/bin:${PATH}"
RUN conda install python=3.8


#########################################################
## Anomalib Development Env
#########################################################

FROM python_base_cuda as anomalib_development_env

# Install all anomalib requirements
COPY ./requirements/requirements.txt /tmp/anomalib/requirements/requirements.txt
RUN pip install -r /tmp/anomalib/requirements/requirements.txt

# Install other requirements related to development
RUN apt-get install -y git
RUN pip install mypy flake8 pytest tox
