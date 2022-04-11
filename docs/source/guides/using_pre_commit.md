(pre-commit_hooks)=
# Pre-commit Hooks

First of all, you will need to install development requirements. This will also install the pre-commit pip package
in your python environment

```pip install -r requirements/dev.txt```

Then, install pre-commit hooks using the following command:

```pre-commit install```

Pre-commit hooks will run several formatters, linters and type checkers each time you commit some changes to a branch. Some tools like black and isort will automatically format your files to enforce consistent formatting within the repo. Other tools will provide a list of errors and warnings which you will be required to address before being able to make the commit.

In some cases it might be desired to commit your changes even though some of the checks are failing. For example when you want to address the pre-commit issues at a later time, or when you want to commit a work-in-progress. In these cases, you can skip the pre-commit hooks by adding the `--no-verify` parameter to the commit command.

```git commit -m 'WIP commit' --no-verify```

When doing so, please make sure to revisit the issues at a later time. A good way to check if all issues have been addressed before making an MR is to run tox.

Apart from tox, you can also run `pre-commit` on all the files to check formatting and style issues. To do this you can use

```pre-commit run --all```
