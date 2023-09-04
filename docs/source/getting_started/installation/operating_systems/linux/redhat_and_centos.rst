.. _redhat_and_centos:

RedHat and CentOS
=================

1. Install Python and Git
-------------------------

You may need to install some additional libraries when using Red Hat,
CentOS, Amazon Linux 2 or Fedora. These steps should work on a clean
install, but please file an Issue if you have any trouble.

.. code:: bash

   sudo yum update
   sudo yum upgrade
   sudo yum install python36-devel mesa-libGL


2. Create a Virtual Environment
-------------------------------

Note: If you already installed openvino-dev and activated the
anomalib_env environment, you can skip to `Step
4 <#4-clone-the-repository>`__. If you use Anaconda, please see the :ref:`conda` installation instructions.

.. code:: bash

   python3 -m venv anomalib_env

3. Activate the Environment
---------------------------

.. code:: bash

   source anomalib_env/bin/activate

4. Clone the Repository
-----------------------

.. code:: bash

   git clone https://github.com/openvinotoolkit/anomalib.git
   cd anomalib

5. Install the Packages
-----------------------

This step installs OpenVINO and dependencies like Jupyter Lab. First,
upgrade pip to the latest version. Then, install the required
dependencies.

.. code:: bash

   python -m pip install --upgrade pip
   pip install -r requirements.txt

Troubleshooting
---------------

-  If you use Anaconda or Miniconda, see the :ref:`conda` installation
   instructions.
