.. _azureml:

Azure ML Studio
===============

The steps below assume that you have an Azure account and access to the
`Azure ML Studio <https://ml.azure.com/>`__. The entire one-time setup
process may take up to 20 minutes.

Step 0: Add a Compute Instance
------------------------------

In Azure ML Studio, `add a compute
instance <https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-manage-compute-instance?tabs=python>`__
and pick any CPU-based instance (No GPU required, but we recommend at
least 4 CPU cores and 8GB of RAM).

Once the compute instance is running, open the terminal and then run
Steps 1-8 below.

Step 1: Create a Conda Environment
----------------------------------

.. code:: bash

   conda create --name anomalib_env python=3.8 -y

Step 2: Activate the Environment
--------------------------------

.. code:: bash

   conda activate anomalib_env

Step 3: Clone OpenVINO Notebooks
--------------------------------

.. code:: bash

   git clone https://github.com/openvinotoolkit/anomalib.git

Step 4: Change Directory to anomalib
----------------------------------------------

.. code:: bash

   cd anomalib

Step 5: Upgrade pip and Install Requirements
--------------------------------------------

.. code:: bash

   python -m pip install --upgrade pip
   pip install .[full]

Step 6: Add anomalib_env to PATH
--------------------------------

.. code:: bash

   set PATH="/anaconda/envs/anomalib_env/bin;%PATH%"
