# Contribution Guidelines

##  {octicon}`light-bulb` Getting Started

To get started with Anomalib development, follow these steps:

### {octicon}`repo-forked` Fork and Clone the Repository

First, fork the Anomalib repository by following the GitHub documentation on [forking a repo](https://docs.github.com/en/enterprise-cloud@latest/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). Then, clone your forked repository to your local machine and create a new branch from `main`.

### {octicon}`gear` Set Up Your Development Environment

Set up your development environment to start contributing. This involves installing the required dependencies and setting up pre-commit hooks for code quality checks. Note that this guide assumes you are using [Conda](https://docs.conda.io/en/latest/) for package management. However, the steps are similar for other package managers.

:::{dropdown} Development Environment Setup Instructions
:icon: gear

1. Create and activate a new Conda environment:

   ```bash
   conda create -n anomalib_dev python=3.10
   conda activate anomalib_dev
   ```

   Install the development requirements:

   ```bash
   # Option I: Via anomalib install
   anomalib install --option dev

   #Option II: Via pip install
   pip install -e .[dev]
   ```

   Optionally, for a full installation with all dependencies:

   ```bash
   # Option I: via anomalib install
   anomalib install --option full

   # Option II: via pip install
   pip install -e .[full]
   ```

2. Install and configure pre-commit hooks:

   ```bash
   pre-commit install
   ```

   Pre-commit hooks help ensure code quality and consistency. After each commit,
   `pre-commit` will automatically run the configured checks for the changed file.
   If you would like to manually run the checks for all files, use:

   ```bash
   pre-commit run --all-files
   ```

   To bypass pre-commit hooks temporarily (e.g., for a work-in-progress commit), use:

   ```bash
   git commit -m 'WIP commit' --no-verify
   ```

   However, make sure to address any pre-commit issues before finalizing your pull request.

:::

## {octicon}`diff` Making Changes

1. **Write Code:** Follow the project's coding standards and write your code with clear intent. Ensure your code is well-documented and includes examples where appropriate. For code quality we use ruff, whose configuration is in [`pyproject.toml`](https://github.com/openvinotoolkit/anomalib/blob/main/pyproject.toml) file.

2. **Add Tests:** If your code includes new functionality, add corresponding tests using [pytest](https://docs.pytest.org/en/7.4.x/) to maintain coverage and reliability.

3. **Update Documentation:** If you've changed APIs or added new features, update the documentation accordingly. Ensure your docstrings are clear and follow [Google's docstring guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

4. **Pass Tests and Quality Checks:** Ensure the test suite passes and that your code meets quality standards by running:

   ```bash
   pre-commit run --all-files
   pytest tests/
   ```

5. **Update the Changelog:** For significant changes, add a summary to the [CHANGELOG](https://github.com/openvinotoolkit/anomalib/blob/main/CHANGELOG.md).

6. **Check Licensing:** Ensure you own the code or have rights to use it, adhering to appropriate licensing.

7. **Sign Your Commits:** Use signed commits to certify that you have the right to submit the code under the project's license:

   ```bash
   git commit -S -m "Your detailed commit message"
   ```

   For more on signing commits, see [GitHub's guide on signing commits](https://docs.github.com/en/github/authenticating-to-github/managing-commit-signature-verification/signing-commits).

## {octicon}`git-pull-request` Submitting Pull Requests

Once you've followed the above steps and are satisfied with your changes:

1. Push your changes to your forked repository.
2. Go to the original Anomalib repository you forked and click "New pull request".
3. Choose your fork and the branch with your changes to open a pull request.
4. Fill in the pull request template with the necessary details about your changes.

We look forward to your contributions!
