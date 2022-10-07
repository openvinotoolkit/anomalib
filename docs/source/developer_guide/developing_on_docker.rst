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

This creates an interactive bash session inside the Docker container, in which you can run all anomalib commands as described in the `readme <https://github.com/openvinotoolkit/anomalib/blob/main/README.md>`_.

The source code is mapped into the running container by means of the `-v "$(pwd)":/anomalib` parameter. This facilitates changes to the source code without having to rebuild the Docker image.

.. note::
	To forward graphical output of the Docker container to the host operating system, you have to disable access control of the host's X11 server. To this end, execute the shell command `xhost +` on the host.


Using VSCode
============

You may also use the Remote-Containers extension for VSCode for deployment with Docker, which can be set up as follows:

1. Install and run `VSCode <https://code.visualstudio.com/download>`_
2. Install the Remote-Containers extension by Microsoft in VSCode.
3. Open command panel by pressing ``CTRL + SHIFT + P``.
4. Type ``Remote-Containers: Open Folder in Container`` and press Enter.
5. Navigate to the root of anomalib folder. (The one with Dockerfile).
6. Give it time to build the image and install the extensions. Grab a |:coffee:| as this will take a while for the first time as the container needs to be built.
7. And voilà |:confetti_ball:| you should now see your folder mounted in the docker container and begin developing. Your changes should automatically be reflected in your host directory.

.. note:: If VSCode throws an error regarding not having sufficient privileges, try restarting the system.
