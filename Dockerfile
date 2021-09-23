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
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh --quiet
RUN bash ~/miniconda.sh -b -p /opt/conda
ENV PATH "/opt/conda/bin:${PATH}"
RUN conda install python=3.8

# Install OpenVINO
ARG OPENVINO_VERSION=l_openvino_toolkit_data_dev_ubuntu20_p_2021.4.689
RUN wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4.1/$OPENVINO_VERSION.tgz --quiet && \
    mkdir -p /opt/intel && tar -xzf $OPENVINO_VERSION.tgz -C /opt/intel && \
    ln -s /opt/intel/$OPENVINO_VERSION/ /opt/intel/openvino_2021 && \
    /opt/intel/openvino_2021/install_dependencies/install_openvino_dependencies.sh -y && \
    /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh && \
    /opt/intel/openvino_2021/bin/setupvars.sh && \
    rm -r $OPENVINO_VERSION.tgz



#########################################################
## Anomalib Development Env
#########################################################

FROM python_base_cuda as anomalib_development_env

# Get MVTec Dataset
# cache datasets first as changes to requirements do no affect this stage
RUN wget ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz --quiet
RUN mkdir -p /tmp/anomalib/datasets/MVTec
RUN tar -xf mvtec_anomaly_detection.tar.xz -C /tmp/anomalib/datasets/MVTec

# Install all anomalib requirements
COPY ./requirements/requirements.txt /tmp/anomalib/requirements/requirements.txt
RUN pip install -r /tmp/anomalib/requirements/requirements.txt

# Install other requirements related to development
COPY ./requirements/requirements_dev.txt /tmp/anomalib/requirements/requirements_dev.txt
RUN pip install -r /tmp/anomalib/requirements/requirements_dev.txt
