"""Custom Help Formatters for Anomalib CLI."""

# Copyright (C) 2023 Intel Corporation
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
    """Get the short description from the docstring.

    Args:
        component (type): The component to get the docstring from

    Returns:
        str: The short description
    """
    if component.__doc__ is None:
        return ""
    docstring = docstring_parser.parse(component.__doc__)
    return docstring.short_description


def get_verbosity_subcommand() -> dict:
    """Parse command line arguments and returns a dictionary of key-value pairs.

    Returns:
        A dictionary containing the parsed command line arguments.

    Examples:
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
    """Return a Markdown object containing the introduction text for Anomalib CLI Guide.

    The introduction text includes a brief description of the guide and links to the Github repository and documentation

    Returns:
        A Markdown object containing the introduction text for Anomalib CLI Guide.
    """
    intro_markdown = (
        "# Anomalib CLI Guide\n\n"
        "Github Repository: [https://github.com/openvinotoolkit/anomalib](https://github.com/openvinotoolkit/anomalib)."
        "\n\n"
        "A better guide is provided by the [documentation](https://anomalib.readthedocs.io/en/latest/index.html)."
    )
    return Markdown(intro_markdown)


def get_verbose_usage(subcommand: str = "train") -> str:
    """Return a string containing verbose usage information for the specified subcommand.

    Args:
    ----
        subcommand (str): The name of the subcommand to get verbose usage information for. Defaults to "train".

    Returns:
    -------
        str: A string containing verbose usage information for the specified subcommand.
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
    r"""Get the cli usage from the docstring.

    Args:
    ----
        component (Optional[object]): The component to get the docstring from

    Returns:
    -------
        Optional[str]: The quick-start guide as Markdown format.

    Example:
    -------
        component.__doc__ = '''
            <Prev Section>

            CLI Usage:
                1. First Step.
                2. Second Step.

            <Next Section>
        '''
        >>> get_cli_usage_docstring(component)
        "1. First Step.\n2. Second Step."
    """
    if component is None or component.__doc__ is None or "CLI Usage" not in component.__doc__:
        return None

    pattern = r"CLI Usage:(.*?)(?=\n{2,}|\Z)"
    match = re.search(pattern, component.__doc__, re.DOTALL)

    if match:
        contents = match.group(1).strip().split("\n")
        return "\n".join([content.strip() for content in contents])
    return None


def render_guide(subcommand: str | None = None) -> list:
    """Render a guide for the specified subcommand.

    Args:
    ----
        subcommand (Optional[str]): The subcommand to render the guide for.

    Returns:
    -------
        list: A list of contents to be displayed in the guide.
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

    Attributes:
    verbosity_level : int
        The level of verbosity for the help output.
    subcommand : str | None
        The subcommand to render the guide for.

    Methods:
    add_usage(usage, actions, *args, **kwargs)
        Add usage information to the help output.
    add_argument(action)
        Add an argument to the help output.
    format_help()
        Format the help output.
    """

    verbosity_dict = get_verbosity_subcommand()
    verbosity_level = verbosity_dict["verbosity"]
    subcommand = verbosity_dict["subcommand"]

    def add_usage(self, usage: str | None, actions: list, *args, **kwargs) -> None:
        """Add usage information to the formatter.

        Args:
        ----
            usage (str | None): A string describing the usage of the program.
            actions (list): An list of argparse.Action objects.
            *args (Any): Additional positional arguments to pass to the superclass method.
            **kwargs (Any): Additional keyword arguments to pass to the superclass method.

        Returns:
        -------
            None
        """
        if self.subcommand in REQUIRED_ARGUMENTS:
            if self.verbosity_level == 0:
                actions = []
            elif self.verbosity_level == 1:
                actions = [action for action in actions if action.dest in REQUIRED_ARGUMENTS[self.subcommand]]

        super().add_usage(usage, actions, *args, **kwargs)

    def add_argument(self, action: argparse.Action) -> None:
        """Add an argument to the help formatter.

        If the verbose level is set to 0, the argument is not added.
        If the verbose level is set to 1 and the argument is not in the non-skip list, the argument is not added.

        Args:
        ----
            action (argparse.Action): The action to add to the help formatter.
        """
        if self.subcommand in REQUIRED_ARGUMENTS:
            if self.verbosity_level == 0:
                return
            if self.verbosity_level == 1 and action.dest not in REQUIRED_ARGUMENTS[self.subcommand]:
                return
        super().add_argument(action)

    def format_help(self) -> str:
        """Format the help message for the current command and returns it as a string.

        The help message includes information about the command's arguments and options,
        as well as any additional information provided by the command's help guide.

        Returns:
            str: A string containing the formatted help message.
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
