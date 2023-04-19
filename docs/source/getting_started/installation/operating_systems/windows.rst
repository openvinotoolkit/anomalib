.. _windows:
Windows
=======

.. warning::
    Skip Steps 1 and 2 if you already installed Python3 and Git on Windows.

1. Install Python
-----------------

.. warning::
    The version of Python that is available in the Microsoft Store is not recommended. It may require installation of additional packages to work well with OpenVINO and the notebooks.

-  Download a Python installer from python.org. Choose Python 3.8,
   3.9 or 3.10 and make sure to pick a 64 bit version. For example, this
   3.8 installer:
   https://www.python.org/ftp/python/3.8.8/python-3.8.8-amd64.exe
-  Double click on the installer to run it, and follow the steps in the
   installer. **Check the box to add Python to your PATH**, and to
   install ``py``. At the end of the installer, there is an option to
   disable the PATH length limit. It is recommended to click this.

2. Install Git
--------------

-  Download `GIT <https://git-scm.com/>`__ from `this
   link <https://github.com/git-for-windows/git/releases/download/v2.35.1.windows.2/Git-2.35.1.2-64-bit.exe>`__
-  Double click on the installer to run it, and follow the steps in the
   installer.

3. Install C++ Redistributable (For Python 3.8)
-----------------------------------------------

-  Download `Microsoft Visual C++
   Redistributable <https://aka.ms/vs/16/release/vc_redist.x64.exe>`__.
-  Double click on the installer to run it, and follow the steps in the
   installer.

4. Install the Notebooks
------------------------

After installing Python 3 and Git, run each step below using *Command
Prompt (cmd.exe)*, not *PowerShell*. Note: If OpenVINO is installed
globally, please do not run any of these commands in a terminal where
setupvars.bat is sourced.

5. Create a Virtual Environment
-------------------------------

If you use Anaconda, please see the :ref:`conda` section.

.. code:: bash

   python -m venv anomalib_env

6. Activate the Environment
---------------------------

.. code:: bash

   anomalib_env\Scripts\activate

7. Clone the Repository
-----------------------

.. code:: bash

   git clone https://github.com/openvinotoolkit/anomalib.git
   cd anomalib

8. Install the Packages
-----------------------

This step installs OpenVINO and dependencies like Jupyter Lab. First,
upgrade pip to the latest version. Then, install the required
dependencies.

.. code:: bash

   python -m pip install --upgrade pip wheel setuptools
   pip install -r .[full]

Troubleshooting
---------------

-  If you have installed multiple versions of Python, use ``py -3.8``
   when creating your virtual environment to specify a supported version
   (in this case 3.7).

-  If you use Anaconda, you may need to add OpenVINO to your Windows
   PATH. See the
   `wiki/Conda <https://github.com/openvinotoolkit/anomalib/wiki/Conda>`__
   page.

-  If you see an error about needing to install C++, please either
   install `Microsoft Visual C++
   Redistributable <https://visualstudio.microsoft.com/downloads/#microsoft-visual-c-redistributable-for-visual-studio-2019>`__
   or use Python 3.7, which does not have this requirement.
