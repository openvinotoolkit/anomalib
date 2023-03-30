.. _ubuntu:

Ubuntu
======

1. Install Python, Git and GPU drivers (optional)
-------------------------------------------------

You may need to install some additional libraries on Ubuntu Linux. These
steps work on a clean install of Ubuntu Desktop 20.04, and should also
work on Ubuntu 22.04 and 20.10, and on Ubuntu Server.

.. code:: bash

   sudo apt-get update
   sudo apt-get upgrade
   sudo apt-get install python3-venv build-essential python3-dev git-all

If you have a CPU with an Intel Integrated Graphics Card, you can
install the `Intel Graphics Compute
Runtime <https://github.com/intel/compute-runtime>`__ to enable
inference on this device. The command for Ubuntu 20.04 is:

   Note: Only execute this command if you do not yet have OpenCL drivers
   installed.

.. code:: bash

   sudo apt-get install intel-opencl-icd

Also, please follow the instructions discussed
`here <https://github.com/openvinotoolkit/anomalib/discussions/540>`__
to ensure you enabled the right permissions.

See the `documentation <https://github.com/intel/compute-runtime>`__ for
other installation methods and instructions for other versions.


2. Create a Virtual Environment
-------------------------------

Note: If you already installed openvino-dev and activated the
anomalib_env environment, you can skip to `Step
4 <#4-clone-the-repository>`__. If you use Anaconda, please see the
`Conda
guide <https://github.com/openvinotoolkit/anomalib/wiki/Conda>`__.

.. code:: bash

   python3 -m venv anomalib_env

3. Activate the Environment
---------------------------

.. code:: bash

   source anomalib_env/bin/activate

4. Clone the Repository
-----------------------

.. code:: bash

   git clone  https://github.com/openvinotoolkit/anomalib.git
   cd anomalib

5. Install the Packages
-----------------------

This step installs OpenVINO and dependencies like Jupyter Lab. First,
upgrade pip to the latest version. Then, install the required
dependencies.

.. code:: bash

   python -m pip install --upgrade pip
   pip install wheel setuptools
   pip install -r requirements.txt


Troubleshooting
---------------

-  The system default version of Python on Ubuntu 20.04 is Python 3.8,
   on Ubuntu 22.04 - 3.10. If you also installed other versions of
   Python, it is recommended to use the full path the to system default
   Python: ``/usr/bin/python3.8 -m venv anomalib_env`` on Ubuntu 20,
   ``/usr/bin/python3.10 -m venv anomalib_env`` on Ubuntu 22.

-  If you use Anaconda or Miniconda, see the :ref:`conda` installation instructions.

-  On Ubuntu, if you see the error **“libpython3.8m.so.1.0: cannot open
   shared object file: No such object or directory”** please install the
   required package using ``apt install libpython3.8-dev``.

-  On Ubuntu, if you see the error **“OSError(‘sndfile library not
   found’) OSError: sndfile library not found”** please install the
   required package using ``apt install libsndfile1``.

-  On Ubuntu, if the GPU device is not found, please follow the
   instruction here to ensure you have installed the drivers and set the
   right permission.
   (https://github.com/openvinotoolkit/openvino_notebooks/discussions/540)
