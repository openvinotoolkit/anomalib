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

By default
`python tools/train.py <https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/blob/samet/stfpm/tools/train.py>`__
runs `STFPM <https://arxiv.org/pdf/2103.04257.pdf>`__ model
`MVTec AD <https://www.mvtec.com/company/research/datasets/mvtec-ad>`__
``leather`` dataset.

::

    python tools/train.py    # Train STFPM on MVTec AD leather

Training a model on a specific dataset and category requires further
configuration. Each model has its own configuration file,
`config.yaml <https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/blob/samet/stfpm/anomalib/models/stfpm/config.yaml>`__,
which contains data, model and training configurable parameters. To
train a specific model on a specific dataset and category, the config
file is to be provided:

::

    python tools/train.py --config <path/to/model/config.yaml>

Alternatively, a model name could also be provided as an argument, where
the scripts automatically finds the corresponding config file.

::

    python tools/train.py --model stfpm

To see a list of currently supported models, refer to page: :ref:`available models`

Development
===========

To setup the development environment, you will need to install development requirements. :code:`pip install -r requirements_dev.txt`

To enforce consistency within the repo, we use several formatters, linters, and style- and type checkers:

.. list-table::
   :widths: 1 1 1
   :header-rows: 1

   * - Tool
     - Function
     - Documentation
   * - Black
     - Code formatting
     - https://black.readthedocs.io/en/stable/
   * - isort
     - Organize import statements
     - https://pycqa.github.io/isort/
   * - Flake8
     - Code style
     - https://flake8.pycqa.org/en/latest/
   * - Pylint
     - Linting
     - http://pylint.pycqa.org/en/latest/
   * - MyPy
     - Type checking
     - https://mypy.readthedocs.io/en/stable/

Instead of running each of these tools manually, we automatically run them before each commit and after each merge request. To achieve this we use pre-commit hooks and tox. Every developer is expected to use pre-commit hooks to make sure that their code remains free of typing and linting issues, and complies with the coding style requirements. When an MR is submitted, tox will be automatically invoked from the CI pipeline in Gitlab to check if the code quality is up to standard. Developers can also run tox locally before making an MR, though this is not strictly necessary since pre-commit hooks should be sufficient to prevent code quality issues. More detailed explanations of how to work with these tools is given in the respective guides:

Pre-commit hooks: :ref:`Pre-commit hooks guide<pre-commit_hooks>`

Tox: :ref:`Using Tox<using_tox>`

In rare cases it might be desired to ignore certain errors or warnings for a particular part of your code. Flake8, Pylint and MyPy allow disabling specific errors for a line or block of code. The instructions for this can be found in the the documentations of each of these tools. Please make sure to only ignore errors/warnings when absolutely necessary, and always add a comment in your code stating why you chose to ignore it.
