"""Anomalib Path Utils."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path


def create_versioned_dir(root_dir: str | Path) -> Path:
    """Create a new version directory and update the ``latest`` symbolic link.

    Args:
        root_dir (Path): The root directory where the version directories are stored.

    Returns:
        latest_link_path (Path): The path to the ``latest`` symbolic link.

    Examples:
        >>> version_dir = create_version_dir(Path('path/to/experiments/'))
        PosixPath('/path/to/experiments/latest')

        >>> version_dir.resolve().name
        v1

        Calling the function again will create a new version directory and
        update the ``latest`` symbolic link:

        >>> version_dir = create_version_dir('path/to/experiments/')
        PosixPath('/path/to/experiments/latest')

        >>> version_dir.resolve().name
        v2

    """
    # Compile a regular expression to match version directories
    version_pattern = re.compile(r"^v(\d+)$")

    # Resolve the path
    root_dir = Path(root_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    # Find the highest existing version number
    highest_version = -1
    for version_dir in root_dir.iterdir():
        if version_dir.is_dir():
            match = version_pattern.match(version_dir.name)
            if match:
                version_number = int(match.group(1))
                highest_version = max(highest_version, version_number)

    # The new directory will have the next highest version number
    new_version_number = highest_version + 1
    new_version_dir = root_dir / f"v{new_version_number}"

    # Create the new version directory
    new_version_dir.mkdir()

    # Update the 'latest' symbolic link to point to the new version directory
    latest_link_path = root_dir / "latest"
    if latest_link_path.is_symlink() or latest_link_path.exists():
        latest_link_path.unlink()
    latest_link_path.symlink_to(new_version_dir, target_is_directory=True)

    return latest_link_path


def convert_to_snake_case(s: str) -> str:
    """Converts a string to snake case.

    Args:
        s (str): The input string to be converted.

    Returns:
        str: The converted string in snake case.

    Examples:
        >>> convert_to_snake_case("Snake Case")
        'snake_case'

        >>> convert_to_snake_case("snakeCase")
        'snake_case'

        >>> convert_to_snake_case("snake_case")
        'snake_case'
    """
    # Replace whitespace, hyphens, periods, and apostrophes with underscores
    s = re.sub(r"\s+|[-.\']", "_", s)

    # Insert underscores before capital letters (except at the beginning of the string)
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

    # Remove leading and trailing underscores
    s = re.sub(r"^_+|_+$", "", s)

    # Replace multiple consecutive underscores with a single underscore
    return re.sub(r"__+", "_", s)


def convert_to_title_case(text: str) -> str:
    """Converts a given text to title case, handling regular text, snake_case, and camelCase.

    Args:
        text (str): The input text to be converted to title case.

    Returns:
        str: The input text converted to title case.

    Raises:
        TypeError: If the input is not a string.

    Examples:
        Regular text:
        >>> convert_to_title_case("the quick brown fox")
        'The Quick Brown Fox'

        Snake case:
        >>> convert_to_title_case("convert_snake_case_to_title_case")
        'Convert Snake Case To Title Case'

        Camel case:
        >>> convert_to_title_case("convertCamelCaseToTitleCase")
        'Convert Camel Case To Title Case'

        Pascal case:
        >>> convert_to_title_case("ConvertPascalCaseToTitleCase")
        'Convert Pascal Case To Title Case'

        Mixed cases:
        >>> convert_to_title_case("mixed_snake_camelCase and PascalCase")
        'Mixed Snake Camel Case And Pascal Case'

        Handling punctuation and contractions:
        >>> convert_to_title_case("what's the_weather_like? it'sSunnyToday.")
        "What's The Weather Like? It's Sunny Today."

        With numbers and special characters:
        >>> convert_to_title_case("python3.9_features and camelCaseNames")
        'Python 3.9 Features And Camel Case Names'
    """
    if not isinstance(text, str):
        msg = "Input must be a string"
        raise TypeError(msg)

    # Handle snake_case
    text = text.replace("_", " ")

    # Handle camelCase and PascalCase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)

    # Split the text into words, preserving punctuation
    words = re.findall(r"[\w']+|[.,!?;]", text)

    # Capitalize each word
    result = [word.capitalize() if word.isalpha() or "'" in word else word for word in words]

    # Join the words back together
    return " ".join(result)


def generate_output_filename(
    input_path: str | Path,
    output_path: str | Path,
    dataset_name: str,
    category: str | None = None,
    mkdir: bool = True,
) -> Path:
    """Generate an output filename based on the input path, preserving the directory structure.

    This function takes an input path, an output base directory, a dataset name, and an optional
    category. It generates an output path that preserves the directory structure after the dataset
    name (and category, if provided) while placing the file in the specified output directory.

    Args:
        input_path (str | Path): The input file path.
        output_path (str | Path): The base output directory.
        dataset_name (str): The name of the dataset in the input path.
        category (str | None, optional): The category name in the input path. Defaults to None.
        mkdir (bool, optional): Whether to create the output directory. Defaults to True.

    Returns:
        Path: The generated output file path.

    Raises:
        ValueError: If the dataset name or category (if provided) is not found in the input path.

    Examples:
        >>> input_path = "/data/MVTec/bottle/test/broken_large/000.png"
        >>> output_base = "/results"
        >>> dataset = "MVTec"

        # With category
        >>> generate_output_filename(input_path, output_base, dataset, "bottle")
        PosixPath('/results/test/broken_large/000.png')

        # Without category
        >>> generate_output_filename(input_path, output_base, dataset)
        PosixPath('/results/bottle/test/broken_large/000.png')

        # Different dataset structure
        >>> input_path = "/datasets/MyDataset/train/class_A/image_001.jpg"
        >>> generate_output_filename(input_path, "/output", "MyDataset", "class_A")
        PosixPath('/output/image_001.jpg')

        # Error case: Dataset not in path
        >>> generate_output_filename("/wrong/path/image.png", "/out", "NonexistentDataset")
        Traceback (most recent call last):
            ...
        ValueError: Dataset name 'NonexistentDataset' not found in the input path.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Find the position of the dataset name in the path
    try:
        dataset_index = input_path.parts.index(dataset_name)
    except ValueError:
        msg = f"Dataset name '{dataset_name}' not found in the input path."
        raise ValueError(msg) from None

    # Determine the start index for preserving subdirectories
    start_index = dataset_index + 1
    if category:
        try:
            category_index = input_path.parts.index(category, dataset_index)
            start_index = category_index + 1
        except ValueError:
            msg = f"Category '{category}' not found in the input path after the dataset name."
            raise ValueError(msg) from None

    # Preserve all subdirectories after the category (or dataset if no category)
    subdirs = input_path.parts[start_index:-1]  # Exclude the filename

    # Construct the output path
    output_path = output_path / Path(*subdirs)

    # Create the output directory if it doesn't exist
    if mkdir:
        output_path.mkdir(parents=True, exist_ok=True)

    # Create and return the output filename
    return output_path / input_path.name
