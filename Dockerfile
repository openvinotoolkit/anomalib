#########################################################
## Python Environment with CUDA
#########################################################

FROM ubuntu:focal AS python_base_cuda
MAINTAINER Anomalib Development Team

# Update system and install wget
RUN apt-get update && apt-get install -y wget ffmpeg libpython3.8 g++

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /opt/conda
ENV PATH "/opt/conda/bin:${PATH}"
RUN conda install python=3.8

# Install CUDA
RUN conda install -c anaconda cudatoolkit==11.3.1
run conda install -c anaconda cudnn

#########################################################
## Anomalib Development Env
#########################################################

FROM python_base_cuda as anomalib_development_env

# Install all anomalib requirements
COPY ./requirements.txt /tmp/anomalib/requirements.txt
RUN pip install -r /tmp/anomalib/requirements.txt

# Install other requirements related to development
RUN apt-get install -y git
RUN pip install mypy flake8 pytest tox