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

::

    yes | conda create -n anomalib python=3.8
    conda activate anomalib
    pip install -r requirements.txt


Optionally, if you prefer using a Docker image for development, refer to the guide :ref:`developing_on_docker`

Training
==============

If you plan on using
`SigOpt <https://app.sigopt.com/docs/runs/get-started>`__ as your
logger, you will need to have that configured prior to running train.py.

By default
`python train.py <https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/blob/samet/stfpm/train.py>`__
runs `STFPM <https://arxiv.org/pdf/2103.04257.pdf>`__ model
`MVTec <https://www.mvtec.com/company/research/datasets/mvtec-ad>`__
``leather`` dataset.

::

    python train.py    # Train STFPM on MVTec leather

Training a model on a specific dataset and category requires further
configuration. Each model has its own configuration file,
`config.yaml <https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/blob/samet/stfpm/anomalib/models/stfpm/config.yaml>`__,
which contains data, model and training configurable parameters. To
train a specific model on a specific dataset and category, the config
file is to be provided:

::

    python train.py --model_config_path <path/to/model/config.yaml>

Alternatively, a model name could also be provided as an argument, where
the scripts automatically finds the corresponding config file.

::

    python train.py --model stfpm

To see a list of currently supported models, refer to page: :ref:`available models`

Development
===========

To setup the development environment, you will need to install development requirements. :code:`pip install -r requirements_dev.txt`

Developers are also required to install pre-commit hooks

::

    pre-commit install

When submitting an MR developers should run Tox. See the :ref:`Using Tox<using_tox>` section for more details.
Submit an MR only when all the checks are passed in Tox.
