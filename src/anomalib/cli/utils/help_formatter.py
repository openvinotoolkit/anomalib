"""Custom help formatters for Anomalib CLI.

This module provides custom help formatting functionality for the Anomalib CLI,
including rich text formatting and customized help output for different verbosity levels.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import sys

import docstring_parser
from jsonargparse import DefaultHelpFormatter
from rich.markdown import Markdown
from rich.panel import Panel
from rich_argparse import RichHelpFormatter

REQUIRED_ARGUMENTS = {
    "train": {"model", "model.help", "data", "data.help", "ckpt_path", "config"},
    "fit": {"model", "model.help", "data", "data.help", "ckpt_path", "config"},
    "validate": {"model", "model.help", "data", "data.help", "ckpt_path", "config"},
    "test": {"model", "model.help", "data", "data.help", "ckpt_path", "config"},
    "predict": {"model", "model.help", "data", "data.help", "ckpt_path", "config"},
    "export": {"model", "model.help", "export_type", "ckpt_path", "config"},
}

try:
    from anomalib.engine import Engine

    DOCSTRING_USAGE = {
        "train": Engine.train,
        "fit": Engine.fit,
        "validate": Engine.validate,
        "test": Engine.test,
        "predict": Engine.predict,
        "export": Engine.export,
    }
except ImportError:
    print("To use other subcommand using `anomalib install`")


def get_short_docstring(component: type) -> str:
    """Get the short description from a component's docstring.

    Args:
        component (type): The component to extract the docstring from.

    Returns:
        str: The short description from the docstring, or empty string if no docstring.

    Example:
        >>> class MyClass:
        ...     '''My class description.
        ...
        ...     More details here.
        ...     '''
        ...     pass

        >>> output = get_short_docstring(MyClass)
        >>> print(output)
        My class description.
    """
    if component.__doc__ is None:
        return ""
    docstring = docstring_parser.parse(component.__doc__)
    return docstring.short_description


def get_verbosity_subcommand() -> dict:
    """Parse command line arguments for verbosity and subcommand.

    Returns:
        dict: Dictionary containing:
            - subcommand: The subcommand being run
            - help: Whether help was requested
            - verbosity: Verbosity level (0-2)

    Example:
        >>> import sys
        >>> sys.argv = ['anomalib', 'train', '-h', '-v']
        >>> get_verbosity_subcommand()
        {'subcommand': 'train', 'help': True, 'verbosity': 1}
    """
    arguments: dict = {"subcommand": None, "help": False, "verbosity": 2}
    if len(sys.argv) >= 2 and sys.argv[1] not in {"--help", "-h"}:
        arguments["subcommand"] = sys.argv[1]
    if "--help" in sys.argv or "-h" in sys.argv:
        arguments["help"] = True
        if arguments["subcommand"] in REQUIRED_ARGUMENTS:
            arguments["verbosity"] = 0
            if "-v" in sys.argv or "--verbose" in sys.argv:
                arguments["verbosity"] = 1
            if "-vv" in sys.argv:
                arguments["verbosity"] = 2
    return arguments


def get_intro() -> Markdown:
    """Get the introduction text for the Anomalib CLI guide.

    Returns:
        Markdown: A Markdown object containing the introduction text with links
            to the Github repository and documentation.

    Example:
        >>> intro = get_intro()
        >>> print(intro)
        # Anomalib CLI Guide
        Github Repository: https://github.com/openvinotoolkit/anomalib
        Documentation: https://anomalib.readthedocs.io/
    """
    intro_markdown = (
        "# Anomalib CLI Guide\n\n"
        "Github Repository: [https://github.com/openvinotoolkit/anomalib](https://github.com/openvinotoolkit/anomalib)."
        "\n\n"
        "A better guide is provided by the [documentation](https://anomalib.readthedocs.io/en/latest/index.html)."
    )
    return Markdown(intro_markdown)


def get_verbose_usage(subcommand: str = "train") -> str:
    """Get verbose usage information for a subcommand.

    This function generates a formatted string containing usage instructions for running
    an Anomalib CLI subcommand with different verbosity levels. The instructions show
    how to access more detailed help information using the -v and -vv flags.

    Args:
        subcommand (str, optional): The subcommand to get usage information for.
            Defaults to "train".

    Returns:
        str: A formatted string containing verbose usage information with examples
            showing different verbosity levels.

    Example:
        Get usage information for the "train" subcommand:

        >>> usage = get_verbose_usage("train")
        >>> print(usage)  # doctest: +NORMALIZE_WHITESPACE
        To get more overridable argument information, run the command below.
        ```python
        # Verbosity Level 1
        anomalib train [optional_arguments] -h -v
        # Verbosity Level 2
        anomalib train [optional_arguments] -h -vv
        ```

        Get usage for a different subcommand:

        >>> usage = get_verbose_usage("export")  # doctest: +NORMALIZE_WHITESPACE
        >>> print(usage)
        To get more overridable argument information, run the command below.
        ```python
        # Verbosity Level 1
        anomalib export [optional_arguments] -h -v
        # Verbosity Level 2
        anomalib export [optional_arguments] -h -vv
        ```
    """
    return (
        "To get more overridable argument information, run the command below.\n"
        "```python\n"
        "# Verbosity Level 1\n"
        f"anomalib {subcommand} [optional_arguments] -h -v\n"
        "# Verbosity Level 2\n"
        f"anomalib {subcommand} [optional_arguments] -h -vv\n"
        "```"
    )


def get_cli_usage_docstring(component: object | None) -> str | None:
    """Extract CLI usage instructions from a component's docstring.

    This function searches for a "CLI Usage:" section in the component's docstring and
    extracts its contents. The section should be delimited by either double newlines
    or the end of the docstring.

    Args:
        component: The object to extract the CLI usage from. Can be None.

    Returns:
        The CLI usage instructions as a string with normalized whitespace, or None if:
        - The component is None
        - The component has no docstring
        - The docstring has no "CLI Usage:" section

    Examples:
        A docstring with CLI usage section:

        >>> class MyComponent:
        ...     '''My component description.
        ...
        ...     CLI Usage:
        ...         1. Run this command
        ...         2. Then this command
        ...
        ...     Other sections...
        ...     '''
        >>> component = MyComponent()
        >>> print(get_cli_usage_docstring(component))
        1. Run this command
        2. Then this command

        A docstring without CLI usage returns None:

        >>> class NoUsage:
        ...     '''Just a description'''
        >>> print(get_cli_usage_docstring(NoUsage()))
        None

        None input returns None:

        >>> print(get_cli_usage_docstring(None))
        None
    """
    if component is None or component.__doc__ is None or "CLI Usage" not in component.__doc__:
        return None

    pattern = r"CLI Usage:(.*?)(?=\n{2,}|\Z)"
    match = re.search(pattern, component.__doc__, re.DOTALL)

    if match:
        contents = match.group(1).strip().split("\n")
        return "\n".join([content.strip() for content in contents])
    return None


def render_guide(subcommand: str | None = None) -> list[Panel | Markdown]:
    """Render a guide for the specified subcommand.

    This function generates a formatted guide containing usage instructions and examples
    for a given CLI subcommand.

    Args:
        subcommand: The subcommand to render the guide for. If None or not found in
            DOCSTRING_USAGE, returns an empty list.

    Returns:
        A list containing rich formatting elements (Panel, Markdown) to be displayed
        in the guide.

    Examples:
        >>> # Empty list for invalid subcommand
        >>> render_guide("invalid")
        []

        >>> # Guide with intro and usage for valid subcommand
        >>> guide = render_guide("train")
        >>> len(guide) > 0
        True

    Notes:
        - The guide includes an introduction section from `get_intro()`
        - For valid subcommands, adds CLI usage from docstrings and verbose usage info
        - Usage is formatted in a Panel with "Quick-Start" title
    """
    if subcommand is None or subcommand not in DOCSTRING_USAGE:
        return []
    contents = [get_intro()]
    target_command = DOCSTRING_USAGE[subcommand]
    cli_usage = get_cli_usage_docstring(target_command)
    if cli_usage is not None:
        cli_usage += f"\n{get_verbose_usage(subcommand)}"
        quick_start = Panel(Markdown(cli_usage), border_style="dim", title="Quick-Start", title_align="left")
        contents.append(quick_start)
    return contents


class CustomHelpFormatter(RichHelpFormatter, DefaultHelpFormatter):
    """A custom help formatter for Anomalib CLI.

    This formatter extends the RichHelpFormatter and DefaultHelpFormatter classes to provide
    a more detailed and customizable help output for Anomalib CLI.

    Args:
        *args: Variable length argument list passed to parent classes.
        **kwargs: Arbitrary keyword arguments passed to parent classes.

    Attributes:
        verbosity_dict (dict): Dictionary containing verbosity level and subcommand.
        verbosity_level (int): The level of verbosity for the help output.
        subcommand (str | None): The subcommand to render the guide for.

    Example:
        >>> from argparse import ArgumentParser
        >>> parser = ArgumentParser(formatter_class=CustomHelpFormatter)
        >>> parser.add_argument('--test')
        >>> help_text = parser.format_help()
        >>> isinstance(help_text, str)
        True

    Note:
        The formatter supports different verbosity levels:
        - Level 0: Shows only quick-start guide
        - Level 1: Shows required arguments
        - Level 2+: Shows all arguments
    """

    verbosity_dict = get_verbosity_subcommand()
    verbosity_level = verbosity_dict["verbosity"]
    subcommand = verbosity_dict["subcommand"]

    def add_usage(self, usage: str | None, actions: list, *args, **kwargs) -> None:
        """Add usage information to the formatter.

        Filters the actions shown in the usage section based on verbosity level
        and required arguments for the current subcommand.

        Args:
            usage: A string describing the usage of the program.
            actions: A list of argparse.Action objects.
            *args: Additional positional arguments passed to parent method.
            **kwargs: Additional keyword arguments passed to parent method.

        Example:
            >>> formatter = CustomHelpFormatter()
            >>> formatter.add_usage("usage:", [], groups=[])
            >>> True  # Method completes without error
            True
        """
        if self.subcommand in REQUIRED_ARGUMENTS:
            if self.verbosity_level == 0:
                actions = []
            elif self.verbosity_level == 1:
                actions = [action for action in actions if action.dest in REQUIRED_ARGUMENTS[self.subcommand]]

        super().add_usage(usage, actions, *args, **kwargs)

    def add_argument(self, action: argparse.Action) -> None:
        """Add an argument to the help formatter.

        Controls which arguments are displayed based on verbosity level and
        whether they are required for the current subcommand.

        Args:
            action: The argparse.Action object to potentially add to the help output.

        Example:
            >>> from argparse import Action, ArgumentParser
            >>> parser = ArgumentParser()
            >>> action = parser.add_argument('--test')
            >>> formatter = CustomHelpFormatter()
            >>> formatter.add_argument(action)
            >>> True  # Method completes without error
            True

        Note:
            - At verbosity level 0, no arguments are shown
            - At verbosity level 1, only required arguments are shown
            - At higher verbosity levels, all arguments are shown
        """
        if self.subcommand in REQUIRED_ARGUMENTS:
            if self.verbosity_level == 0:
                return
            if self.verbosity_level == 1 and action.dest not in REQUIRED_ARGUMENTS[self.subcommand]:
                return
        super().add_argument(action)

    def format_help(self) -> str:
        """Format the complete help message.

        Generates a formatted help message that includes command arguments, options,
        and additional guide information based on the current verbosity level.

        Returns:
            str: The formatted help message as a string.

        Example:
            >>> formatter = CustomHelpFormatter()
            >>> help_text = formatter.format_help()
            >>> isinstance(help_text, str)
            True

        Note:
            The output format depends on verbosity level:
            - Level 0-1: Shows quick-start guide for supported subcommands
            - Level 1+: Includes argument section in a panel
            - All levels: Maintains consistent spacing and formatting
        """
        with self.console.capture() as capture:
            section = self._root_section
            if self.subcommand in REQUIRED_ARGUMENTS and self.verbosity_level in {0, 1} and len(section.rich_items) > 1:
                contents = render_guide(self.subcommand)
                for content in contents:
                    self.console.print(content)
            if self.verbosity_level > 0:
                if len(section.rich_items) > 1:
                    section = Panel(section, border_style="dim", title="Arguments", title_align="left")
                self.console.print(section, highlight=False, soft_wrap=True)
        help_msg = capture.get()

        if help_msg:
            help_msg = self._long_break_matcher.sub("\n\n", help_msg).rstrip() + "\n"
        return help_msg
