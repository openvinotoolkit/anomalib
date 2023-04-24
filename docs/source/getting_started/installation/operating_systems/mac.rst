MacOS
=====

.. note::
    Alternatively, skip steps 1-3 if you prefer to manually install Python 3 and Git.

1. Install Xcode Command Line Tools
-----------------------------------

.. code:: bash

   xcode-select --install

2. Install Homebrew
-------------------

.. code:: bash

   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

After you install it, follow the instructions from the Homebrew
installation to set it up.

3. Install Python and dependencies
----------------------------------

.. code:: bash

   brew install python@3.9
   brew install protobuf

Run each step below in a terminal. Note: If OpenVINO is installed
globally, please do not run any of these commands in a terminal where
setupvars.sh is sourced.

4. Create a Virtual Environment
-------------------------------

Note: If you already installed openvino-dev and activated the
anomalib_env environment, you can skip to `Step
6 <#6-clone-the-repository>`__. If you use Anaconda, please see the :ref:`conda` installation instructions.

.. code:: bash

   python3 -m venv anomalib_env

5. Activate the Environment
---------------------------

.. code:: bash

   source anomalib_env/bin/activate

6. Clone the Repository
-----------------------

.. code:: bash

   git clone https://github.com/openvinotoolkit/anomalib.git
   cd anomalib

7. Install the Packages
-----------------------

This step installs OpenVINO and dependencies like Jupyter Lab. First,
upgrade pip to the latest version. Then, install the required
dependencies.

.. code:: bash

   python -m pip install --upgrade pip wheel setuptools
   pip install .[full]


Troubleshooting
---------------

-  If you use Anaconda or Miniconda, see the [[Conda]] wiki page.
