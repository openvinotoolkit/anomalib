.. _sagemaker:

Amazon SageMaker Studio
=======================

The steps below assume that you have an `AWS
account <https://console.aws.amazon.com/console/home?nc2=h_ct&src=header-signin>`__
and access to `Amazon SageMaker
Studio <https://aws.amazon.com/sagemaker/studio/>`__. The entire
one-time setup process may take up to 15 minutes.

Pre-requisites: Launch Amazon SageMaker Studio Environment.
-----------------------------------------------------------

-  Log into your Amazon SageMaker Studio Environment and ``Add user``

-  Choose desired user profile name

-  Choose Jupyter Lab version 3.0

-  Choose the remaining default setting and click Submit to Add user.

-  Click “Open Studio” to Launch the Amazon SageMaker Studio
   environment.

.. note::
    The Amazon SageMaker free tier usage per month for the first 2 months is 250 hours of ml.t3.medium instance on Studio notebook. In this example, we are using an ml.t3.medium instance.

-  Allow a couple of minutes for your environment to spin up. You should
   see the following loading screen:

-  Then, Choose ``Data Science 3.0`` in “select a SageMaker image”
   drop-down under **Notebooks and compute resources**

-  Then, Click on ``**+**`` on ``Image Terminal`` to open a terminal
   session:

Anomalib Setup.
---------------

-  Inside the terminal, follow the steps below.

Step 1: Install few system dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

   apt update
   apt install build-essential -y
   apt install libpython3.9-dev -y
   apt install libgl1-mesa-glx -y

Step 2: Setup OpenVINO conda environment.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

   conda create --name anomalib_env python=3.9
   conda activate anomalib_env
   set PATH="/anaconda/envs/anomalib_env/bin;%PATH%"

Step 3: Setup Anomalib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

   git clone https://github.com/openvinotoolkit/anomalib.git
   cd anomalib
   # Install OpenVINO and OpenVINO notebook Requirements
   python -m pip install --upgrade pip
   pip install .[full]
