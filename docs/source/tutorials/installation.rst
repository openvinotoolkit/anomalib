Installation
===============

The repo is thoroughly tested based on the following configuration.

* Ubuntu 20.04

* NVIDIA GeForce RTX 3090

You will need
`Anaconda <https://www.anaconda.com/products/individual>`__ installed on
your system before proceeding with the Anomaly Library install.

After downloading the Anomaly Library, extract the files and navigate to
the extracted location.

To perform a development install, run the following:

.. code-block:: bash

    yes | conda create -n anomalib python=3.8
    conda activate anomalib
    pip install -r requirements.txt


Optionally, if you prefer using a Docker image for development, refer to the guide :ref:`developing_on_docker`
