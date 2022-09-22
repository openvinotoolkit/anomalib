Getting Started with Development
================================

To setup the development environment, you will need to install development requirements. :code:`pip install -r requirements/dev.txt`

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
