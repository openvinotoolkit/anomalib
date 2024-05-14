# Test Documentation for Anomalib

This guide provides instructions on how to run the tests for Anomalib
using either `pytest` or `tox`. Running tests is crucial for verifying the
functionality and stability of the framework.

## Prerequisites

Before running the tests, make sure you have the following prerequisites
installed:

- Python 3.8 or newer
- `pre-commit`
- `pytest`
- `pytest-cov`
- `pytest-sugar`
- `pytest-xdist`
- `coverage[toml]`
- `tox`

## Setup

It is recommended to run Python projects in a virtual environment to manage
dependencies easily.

To create and activate a virtual environment:

```bash
conda create -n anomalib_env python=3.10
conda activate anomalib_env

# Or using your favorite virtual environment
# ...

# Clone the repository and install in editable mode
git clone https://github.com/openvinotoolkit/anomalib.git
cd anomalib
pip install -e .[all]
```

`.[all]` will install all the dependencies. Alternatively, test pre-requisites
can be installed via `pip install -e requirements/dev.txt`.

## Running Tests with `pytest`

`pytest` is a powerful testing tool that allows for detailed test execution.

To run all tests with pytest, execute:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/path/to/test_file.py
```

For verbose output:

```bash
pytest -v
```

### Checking Test Coverage with `pytest-cov

`pytest-cov` is a plugin for pytest that provides coverage reporting. You can
use it to measure the code coverage of your tests. Here's how to run it:

```bash
# To run pytest with coverage
pytest --cov=./
```

This command will run all tests and report the coverage of your project.
If you want to generate a coverage report in HTML format, you can use the
`--cov-report` option:

To generate a coverage report in HTML format

```bash
pytest --cov=./ --cov-report html
```

After running this command, you will find a new directory named `htmlcov`. Open
the `index.html` file in this directory to view the coverage report.
This will provide instructions on how to check test coverage with `pytest-cov.`

## Running Tests with `tox`

`tox` is a tool that automates testing across multiple Python environments,
ensuring compatibility. In Anomalib, `tox` is used to automate multiple tests.
Here's a brief explanation of each:

- `pre-commit`: These tests are run before each commit is made to the
  repository. They're used to catch issues early and prevent problematic code
  from being committed to the repository. They can include things like style
  checks, unit tests, and static analysis.

- `unit`: These tests are run before code is merged into a main or release
  branch. These are used to test individual components.

- `integration`: These tests are also run before the code is merged into a main
  or release branch. These test whether the components work together. These focus
  mainly on training, exporting and inferring models via the API and the CLI.

- `trivy-scan`: Trivy is a comprehensive, open-source vulnerability scanner for
  containers. A `trivy-scan` would check your project's dependencies for known
  security vulnerabilities.

- `bandit-scan`: Bandit is a tool designed to find common security issues in
  Python code. It would check your code for potential security vulnerabilities.

To run these tests, you can use the `tox` command followed by the test
environment. Here are some examples:

To run the pre-commit tests

```bash
tox -e pre-commit
```

To run the unit and integration tests

```bash
tox -e pre-merge
```

To run the trivy-scan

```bash
tox -e trivy-scan
```

To run the bandit-scan

```bash
tox -e bandit-scan
```

### Checking Test Coverage with `tox`

`tox` in Anomalib is configured to run the pytest coverage. This means that if
the tests are run via `tox`, the coverage report is automatically generated.
