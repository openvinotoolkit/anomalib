"""Anomalib install subcommand code."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from anomalib.cli.utils.installation import (
    get_requirements,
    get_torch_install_args,
    parse_requirements,
)

logger = logging.getLogger("pip")
logger.setLevel(logging.WARNING)  # setLevel: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
console = Console()
handler = RichHandler(
    console=console,
    show_level=False,
    show_path=False,
)
logger.addHandler(handler)


def anomalib_install(option: str = "full", verbose: bool = False) -> int:
    """Install Anomalib requirements.

    Args:
        option (str | None): Optional-dependency to install requirements for.
        verbose (bool): Set pip logger level to INFO

    Raises:
        ValueError: When the task is not supported.

    Returns:
        int: Status code of the pip install command.
    """
    from pip._internal.commands import create_command

    options = (
        [option]
        if option != "full"
        else [option.stem for option in Path("requirements").glob("*.txt") if option.stem != "dev"]
    )
    requirements = get_requirements(requirement_files=options)

    # Parse requirements into torch, mmcv and other requirements.
    # This is done to parse the correct version of torch (cpu/cuda).
    torch_requirement, other_requirements = parse_requirements(requirements, skip_torch="engine" not in options)

    # Get install args for torch to install it from a specific index-url
    install_args: list[str] = []
    torch_install_args = []
    if "engine" in options and torch_requirement is not None:
        torch_install_args = get_torch_install_args(torch_requirement)

    # Combine torch and other requirements.
    install_args = other_requirements + torch_install_args

    # Install requirements.
    with console.status("[bold green]Working on installation...\n") as status:
        if verbose:
            logger.setLevel(logging.INFO)
            status.stop()
        console.log(f"Installation list: [yellow]{install_args}[/yellow]")
        status_code = create_command("install").main(install_args)
        if status_code == 0:
            console.log(f"Installation Complete: {install_args}")

    if status_code == 0:
        console.print("Anomalib Installation [bold green]Complete.[/bold green]")

    return status_code
