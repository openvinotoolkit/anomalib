.. _pre-commit_hooks:

Pre-commit Hooks
================

To enforce consistency within the repo, Anomalib relies on several formatters, linters, and style- and type checkers:

| Tool         | Function                   | Documentation                                  |
| ------       | -------------------------- | -----------------------------------------      |
| Black        | Code formatting            | <https://black.readthedocs.io/en/stable/>      |
| isort        | Organize import statements | <https://pycqa.github.io/isort/>               |
| Flake8       | Code style                 | <https://flake8.pycqa.org/en/latest/>          |
| Pylint       | Linting                    | <http://pylint.pycqa.org/en/latest/>           |
| MyPy         | Type checking              | <https://mypy.readthedocs.io/en/stable/>       |
| Pydocstyle   | Docstring conventions      | <http://www.pydocstyle.org/en/stable/>         |
| Prettier     | Opinionated code formatter | <https://prettier.io/docs/en/index.html>       |
| Markdownlint | Markdown style             | <https://github.com/markdownlint/markdownlint> |
| Hadolint     | Dockerfile linting         | <https://hadolint.github.io/hadolint/>         |

Instead of running each of these tools manually, we automatically run them before each commit and after each merge request. To achieve this we use pre-commit hooks. Every developer is expected to use pre-commit hooks to make sure that their code remains free of typing and linting issues, and complies with the coding style requirements.

To install the development environment, including pre-commit, run the following commands:

.. code-block:: bash

  $ conda create -n anomalib_dev python=3.8
  $ conda activate anomalib_dev
  $ pip install -r requirements/base.txt -r requirements/dev.txt
  $ pre-commit install

Pre-commit hooks will run each of the code quality tools listed above each time you commit some changes to a branch. Some tools like black and isort will automatically format your files to enforce consistent formatting within the repo. Other tools will provide a list of errors and warnings which you will be required to address before being able to make the commit.

In some cases it might be desired to commit your changes even though some of the checks are failing. For example when you want to address the pre-commit issues at a later time, or when you want to commit a work-in-progress. In these cases, you can skip the pre-commit hooks by adding the ``--no-verify`` parameter to the commit command.

.. code-block:: bash

  $ git commit 'WIP commit' --no-verify

When doing so, please make sure to revisit the pre-commit issues before you submit your PR. A good way to confirm if your code passes the checks is by running `pre-commit run --all-files`.

In rare cases it might be desired to ignore certain errors or warnings for a particular part of your code. Flake8, Pylint and MyPy allow disabling specific errors for a line or block of code. The instructions for this can be found in the the documentations of each of these tools. Please make sure to only ignore errors/warnings when absolutely necessary, and always add a comment in your code stating why you chose to ignore it.
