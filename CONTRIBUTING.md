# Contributing to Anomalib

We welcome your input! 👐

We want to make it as simple and straightforward as possible to contribute to this project, whether it is a:

- Bug Report
- Discussion
- Feature Request
- Creating a Pull Request (PR)
- Becoming a maintainer

## Bug Report

We use GitHub issues to track the bugs. Report a bug by using our Bug Report Template in [Issues](https://github.com/openvinotoolkit/anomalib/issues/new?assignees=&labels=&template=bug_report.md).

## Discussion

We enabled [GitHub Discussions](https://github.com/openvinotoolkit/anomalib/discussions/) in anomalib to welcome the community to ask questions and/or propose ideas/solutions. This will not only provide a medium to the community to discuss about anomalib but also help us de-clutter [Issues](https://github.com/openvinotoolkit/anomalib/issues/new?assignees=&labels=&template=bug_report.md).

## Feature Request

We utilize GitHub issues to track the feature requests as well. If you are certain regarding the feature you are interested and have a solid proposal, you could then create the feature request by using our [Feature Request Template](https://github.com/openvinotoolkit/anomalib/issues/new?assignees=&labels=&template=feature_request.md) in Issues. If it's still in an idea phase, you could then discuss that with the community in our [Discussion](https://github.com/openvinotoolkit/anomalib/discussions/categories/ideas).

## Development & PRs

We actively welcome your pull requests:

1. Fork the repo and create your branch from [`main`](https://github.com/openvinotoolkit/anomalib/tree/main).
1. If you've added code that should be tested, add tests.
1. If you've changed APIs, update the documentation.
1. Ensure the test suite passes.
1. Make sure your code lints.
1. Make sure you own the code you're submitting or that you obtain it from a source with an appropriate license.
1. Issue that pull request!

To setup the development environment, you will need to install development requirements.

```bash
pip install -r requirements/dev.txt
```

To enforce consistency within the repo, we use several formatters, linters, and style- and type checkers:

| Tool   | Function                   | Documentation                           |
| ------ | -------------------------- | --------------------------------------- |
| Black  | Code formatting            | https://black.readthedocs.io/en/stable/ |
| isort  | Organize import statements | https://pycqa.github.io/isort/          |
| Flake8 | Code style                 | https://flake8.pycqa.org/en/latest/     |
| Pylint | Linting                    | http://pylint.pycqa.org/en/latest/      |
| MyPy   | Type checking              | https://mypy.readthedocs.io/en/stable/  |

Instead of running each of these tools manually, we automatically run them before each commit and after each merge request. To achieve this we use pre-commit hooks and tox. Every developer is expected to use pre-commit hooks to make sure that their code remains free of typing and linting issues, and complies with the coding style requirements. When an MR is submitted, tox will be automatically invoked from the CI pipeline in Gitlab to check if the code quality is up to standard. Developers can also run tox locally before making an MR, though this is not strictly necessary since pre-commit hooks should be sufficient to prevent code quality issues. More detailed explanations of how to work with these tools is given in the respective guides:

- Pre-commit hooks: [Pre-commit hooks guide](https://openvinotoolkit.github.io/anomalib/guides/using_pre_commit.html#pre-commit-hooks)
- Tox: [Using Tox](https://openvinotoolkit.github.io/anomalib/guides/using_tox.html#using-tox)

In rare cases it might be desired to ignore certain errors or warnings for a particular part of your code. Flake8, Pylint and MyPy allow disabling specific errors for a line or block of code. The instructions for this can be found in the the documentations of each of these tools. Please make sure to only ignore errors/warnings when absolutely necessary, and always add a comment in your code stating why you chose to ignore it.

## License

You accept that your contributions will be licensed under the [Apache-2.0 License](https://choosealicense.com/licenses/apache-2.0/) if you contribute to this repository. If this is a concern, please notify the maintainers.

## References

This document was adapted from [here](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62).
