.. _conda:

Conda
=====

The steps to use the OpenVINO notebooks with Anaconda are slightly
different than with Python from an installer version. This is a modified
installation guide for Anaconda. It has been tested with Miniconda on
Windows 10. If you run into an issue with these steps, please `let us
know <https://github.com/openvinotoolkit/anomalib/discussions>`__.

On Windows, these steps should be executed inside an **Anaconda Prompt**
(open Anaconda Prompt from the start menu, or press Windows-S and start
typing *Anaconda*). Use the regular Anaconda Prompt, not the Powershell
prompt.

Step 1: Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git clone https://github.com/openvinotoolkit/anomalib.git

Step 2: Create a Conda Environment with Python 3.8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this tutorial, we use Python 3.8. If you want to use a higher version, that is also possible.

.. code:: bash

   cd anomalib
   conda create -n anomalib_env python=3.8

Step 3: Activate the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   conda activate anomalib_env

Step 4: Install the Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install OpenVINO, Jupyter, and other required packages to run the
notebooks.

.. code:: bash

   # Upgrade pip to the latest version to ensure compatibility with all dependencies
   python -m pip install --upgrade pip
   pip install .[full]

Step 5 [Optional]: Add the OpenVINO library to your PATH
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
    This step is only for Windows. Skip this step for macOS and Linux.

Depending on your Conda installation, this step may be required at the
moment for conda environments on Windows. In a future OpenVINO version
this should no longer be necessary.

The command below assumes that Miniconda is installed in the default
location: ``C:\Users\<username>\Minoconda3``, where ``<username>`` is
your Windows username. If you installed Anaconda, replace Miniconda3
with Anaconda3. If you installed Anaconda or Minoconda in a different
location, you can run
``python -c "import os,sys;print(os.path.dirname(sys.executable))"`` to
find the path to ``anomalib_env``.

Note that at the moment you need to run this command again if you open a
new Anaconda Prompt to run the notebooks. You can add this folder to
your PATH for every command prompt you open, by following `these
steps <https://docs.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v=office.14)>`__
on Microsoft’s website. Note however, that this may cause issues if you
have multiple OpenVINO versions installed.

.. code:: bash

   set PATH=C:\Users\<username>\Miniconda3\envs\anomalib_env\Lib\site-packages\openvino\libs;%PATH%
