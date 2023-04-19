Installation
============

The `Anomalib README <https://github.com/openvinotoolkit/anomalib#getting-started>`_ provides instructions for how to install and get started with the notebooks. These instructions assume that you have Python and Git installed already, and that Python is installed with a system installer. In this section, you can find more detailed installation guides. If you run into an issue, please feel free to open a `discussion topic <https://github.com/openvinotoolkit/anomalib/discussions>`_!

The following code block shows the installation process in a nutshell. If you want to learn more about the installation process, you can skip to the next section.

.. code-block:: bash

    yes | conda create -n anomalib python=3.8
    conda activate anomalib
    pip install .[full]


For a detailed installation guide, please refer to below:

.. toctree::
   :maxdepth: 2
   :name: start
   :caption: Installation Guide

   operating_systems/index
   environment_managers/index
   cloud_services/index
   docker/index
