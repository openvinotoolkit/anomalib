.. _developing_on_docker:

Developing on Docker
======================

.. note:: Assumes you have GPU on your system with the correct drivers installed

1.	Install `VSCode <https://code.visualstudio.com/download>`_
2.	Install `Docker <https://docs.docker.com/engine/install/>`_
3.	Install `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_

.. warning:: Make sure your user is added to the docker group. First check by typing ``groups`` in terminal. If     ``docker`` is not listed, then use the command ``sudo usermod -aG docker $USER``.

4.	Logout and log back in to ensure that user group is evaluated.
5.	Open VSCode.
6.	Install the Remote-Containers extension by Microsoft in VSCode.
7.	Open command panel by pressing ``CTRL + SHIFT + P``.
8.	Type ``Remote-Containers: Open Folder in Container`` and press Enter.
9.	Navigate to the root of anomalib folder. (The one with Dockerfile).
10.	Give it time to build the image and install the extensions. Grab a |:coffee:| as this will take a while for the first time as the container needs to be built.
11.	And voilà |:confetti_ball:| you should now see your folder mounted in the docker container and begin developing. Your changes should automatically be reflected in your host directory.

.. note:: If VSCode throws an error regarding not having sufficient privileges, try restarting the system.
