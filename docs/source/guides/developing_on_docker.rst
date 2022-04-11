.. _developing_on_docker:

Developing on Docker
======================

.. note:: 
	You need a CUDA-capable GPU with suitable drivers installed

1. Install `Docker <https://docs.docker.com/engine/install/>`_
2. Install `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_


Build the Docker Image
======================

To use anomalib with Docker, you can build a Docker image containing anomalib all its dependencies from the provided Dockerfile. To this end, navigate to the anomalib root directory (the one containing the Dockerfile) and build the Docker image with

.. code-block:: console

	docker build . --tag=anomalib


Run the Docker Image
====================

After building the image, you can run it as follows

.. code-block:: console

	docker run \
	-it --rm \
	--ipc=host \
	--env="DISPLAY" \
	--gpus=all \
	-w /anomalib \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v "$(pwd)":/anomalib \
	anomalib \
	/bin/bash

This creates an interactive bash session inside the Docker container, in which you can run all anomalib commands as described in the `readme <https://github.com/openvinotoolkit/anomalib/blob/development/README.md>`_.

The source code is mapped into the running container by means of the `-v "$(pwd)":/anomalib` parameter. This facilitates changes to the source code without having to rebuild the Docker image.

.. note:: 
	To forward graphical output of the Docker container to the host operating system, you have to disable access control of the host's X11 server. To this end, execute the shell command `xhost +` on the host.
