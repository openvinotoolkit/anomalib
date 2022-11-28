.. _using_tox:

Using Tox
=========

The quality of code for ``anomalib`` maintained by using tools such as ``black``, ``mypy``, and ``flake8``. In order to enforce these, we use Tox. This is also the reason why we don't have dependencies such as ``pytest`` in the ``requirements.txt`` file.

What is tox?::

  ``tox`` aims to automate and standardize testing in Python. It is part of a
  larger vision of easing the packaging, testing and release process of Python
  software.

  It is a generic virtualenv management and test command line tool you can use for:

  * checking that your package installs correctly with different Python versions and interpreters
  * running your tests in each of the environments, configuring your test tool of choice
  * acting as a frontend to Continuous Integration servers, greatly reducing boilerplate and merging CI and shell-based testing. - from the [docs](https://tox.readthedocs.io/)


Setting-up and Getting Started
------------------------------

Setting up tox is easy.

1. Create new Conda environment

.. code-block:: bash

  $ conda create -n toxenv python=3.8

2. Activate the environment

.. code-block:: bash

  $ conda activate toxenv

3. Install tox

.. code-block:: bash

  $ pip install tox

4. Run

``tox``
It should setup the environments and install the dependencies.


.. note::

  All developers are required to run tox before creating their PR. If you have setup pre-commit hooks then this step can be skipped.

Brief Explanation of ``tox.ini``
--------------------------------

Here is a brief explanation of ``tox.ini`` for those who are interested. However, for more details, the readers are directed to the [official docs](https://tox.readthedocs.io/).

All ``tox`` files start with
``[tox]``

``isolated_build = True`` tells tox to use a virtual environment instead of using the global python.

``skip_missing_interpreters = True`` allows tox to run even if some specified environments are missing.

``envlist`` This defines all the environments that tox should create. Take for example that we want to create an environment to run ``black`` formatter. We can name this environment as:

.. code-block:: ini

  [tox]
  envlist =
      black_env

Then to define the commands in this environment all we need to do is start a block with `[testenv:black_env]` and then define the dependencies and call the commands.

.. code-block:: ini

  [testenv:black_env]
      deps = black
      commands = black .

For a more elaborate setup, refer to the `tox.ini` file in this repository.

Useful Tox CheatSheet
---------------------

* `tox -e envname[,name1,name2]` - to run only those environments
* `tox -r [envname..]` - to recreate all/specific environments. Handy when the requirements have changed
* `tox -l` - to list all environments
