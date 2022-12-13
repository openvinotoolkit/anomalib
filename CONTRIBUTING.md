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
1. Set up the development environment by following the instructions below.
1. If you've added code that should be tested, add tests.
1. If you've changed APIs, update the documentation.
1. Ensure the test suite passes.
1. Make sure you own the code you're submitting or that you obtain it from a source with an appropriate license.
1. Add a summary of the changes to the [CHANGELOG](https://github.com/openvinotoolkit/anomalib/blob/main/CHANGELOG.md) (not for minor changes, docs and tests).
1. Issue that pull request!

### Setting up the development environment and using pre-commit

To setup the development environment, you will need to install development requirements, as well as the base requirements of the library.

```bash
conda create -n anomalib_dev python=3.8
conda activate anomalib_dev
pip install -r requirements/base.txt -r requirements/dev.txt
pre-commit install
```

The commands above will install pre-commit. Pre-commit hooks will run each of the code quality tools listed above each time you commit some changes to a branch. Some tools like black and isort will automatically format your files to enforce consistent formatting within the repo. Other tools will provide a list of errors and warnings which you will be required to address before being able to make the commit.

The pre-commit checks can also be triggered manually with the following command:

```bash
pre-commit run --all-files
```

In some cases it might be desired to commit your changes even though some of the checks are failing. For example when you want to address the pre-commit issues at a later time, or when you want to commit a work-in-progress. In these cases, you can skip the pre-commit hooks by adding the `--no-verify` parameter to the commit command.

```bash
git commit -m 'WIP commit' --no-verify
```

When doing so, please make sure to revisit the pre-commit issues before you submit your PR. A good way to confirm if your code passes the checks is by running `pre-commit run --all-files`.

More information on pre-commit and how it is used in Anomalib can be found in our documentation:

- [Pre-commit hooks guide](https://openvinotoolkit.github.io/anomalib/developer_guide/pre_commit_hooks.html)

## License

You accept that your contributions will be licensed under the [Apache-2.0 License](https://choosealicense.com/licenses/apache-2.0/) if you contribute to this repository. If this is a concern, please notify the maintainers.

## References

This document was adapted from [here](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62).
