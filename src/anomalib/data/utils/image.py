"""Image utilities for reading, writing and processing images.

This module provides various utility functions for handling images in Anomalib:

- Reading images in various formats (RGB, grayscale, depth)
- Writing images to disk
- Converting between different image formats
- Processing images (padding, resizing etc.)
- Handling image filenames and paths

Example:
    >>> from anomalib.data.utils import read_image
    >>> # Read image as numpy array
    >>> image = read_image("image.jpg")
    >>> print(type(image))
    <class 'numpy.ndarray'>

    >>> # Read image as tensor
    >>> image = read_image("image.jpg", as_tensor=True)
    >>> print(type(image))
    <class 'torch.Tensor'>
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch.nn import functional as F  # noqa: N812
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms.v2.functional import to_dtype, to_image
from torchvision.tv_tensors import Mask

from anomalib.data.utils.path import validate_path

logger = logging.getLogger(__name__)


def is_image_file(filename: str | Path) -> bool:
    """Check if the filename has a valid image extension.

    Args:
        filename (str | Path): Path to file to check

    Returns:
        bool: ``True`` if filename has valid image extension

    Examples:
        >>> is_image_file("image.jpg")
        True

        >>> is_image_file("image.png")
        True

        >>> is_image_file("image.txt")
        False
    """
    filename = Path(filename)
    return filename.suffix.lower() in IMG_EXTENSIONS


def get_image_filename(filename: str | Path) -> Path:
    """Get validated image filename.

    Args:
        filename (str | Path): Path to image file

    Returns:
        Path: Validated path to image file

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is not an image

    Examples:
        >>> get_image_filename("image.jpg")
        PosixPath('image.jpg')

        >>> get_image_filename("missing.jpg")
        Traceback (most recent call last):
            ...
        FileNotFoundError: File not found: missing.jpg

        >>> get_image_filename("text.txt")
        Traceback (most recent call last):
            ...
        ValueError: ``filename`` is not an image file: text.txt
    """
    filename = Path(filename)

    if not filename.exists():
        msg = f"File not found: {filename}"
        raise FileNotFoundError(msg)

    if not is_image_file(filename):
        msg = f"``filename`` is not an image file: {filename}"
        raise ValueError(msg)
    return filename


def get_image_filenames_from_dir(path: str | Path) -> list[Path]:
    """Get list of image filenames from directory.

    Args:
        path (str | Path): Path to directory containing images

    Returns:
        list[Path]: List of paths to image files

    Raises:
        ValueError: If path is not a directory or no images found

    Examples:
        >>> get_image_filenames_from_dir("images/")
        [PosixPath('images/001.jpg'), PosixPath('images/002.png')]

        >>> get_image_filenames_from_dir("empty/")
        Traceback (most recent call last):
            ...
        ValueError: Found 0 images in empty/
    """
    path = Path(path)
    if not path.is_dir():
        msg = f"Path is not a directory: {path}"
        raise ValueError(msg)

    image_filenames = [get_image_filename(f) for f in path.glob("**/*") if is_image_file(f)]

    if not image_filenames:
        msg = f"Found 0 images in {path}"
        raise ValueError(msg)

    return sorted(image_filenames)


def get_image_filenames(path: str | Path, base_dir: str | Path | None = None) -> list[Path]:
    """Get list of image filenames from path.

    Args:
        path (str | Path): Path to image file or directory
        base_dir (str | Path | None): Base directory to restrict file access

    Returns:
        list[Path]: List of paths to image files

    Examples:
        >>> get_image_filenames("image.jpg")
        [PosixPath('image.jpg')]

        >>> get_image_filenames("images/")
        [PosixPath('images/001.jpg'), PosixPath('images/002.png')]

        >>> get_image_filenames("images/", base_dir="allowed/")
        Traceback (most recent call last):
            ...
        ValueError: Access denied: Path is outside the allowed directory
    """
    path = validate_path(path, base_dir)
    image_filenames: list[Path] = []

    if path.is_file():
        image_filenames = [get_image_filename(path)]
    elif path.is_dir():
        image_filenames = get_image_filenames_from_dir(path)
    else:
        msg = "Path is not a file or directory"
        raise FileNotFoundError(msg)

    return image_filenames


def duplicate_filename(path: str | Path) -> Path:
    """Add numeric suffix to filename if it already exists.

    Args:
        path (str | Path): Path to file

    Returns:
        Path: Path with numeric suffix if original exists

    Examples:
        >>> duplicate_filename("image.jpg")  # File doesn't exist
        PosixPath('image.jpg')

        >>> duplicate_filename("exists.jpg")  # File exists
        PosixPath('exists_1.jpg')

        >>> duplicate_filename("exists.jpg")  # Both exist
        PosixPath('exists_2.jpg')
    """
    path = Path(path)

    if not path.exists():
        return path

    i = 0
    while True:
        duplicated_path = path if i == 0 else path.parent / (path.stem + f"_{i}" + path.suffix)
        if not duplicated_path.exists():
            break
        i += 1

    return duplicated_path


def generate_output_image_filename(input_path: str | Path, output_path: str | Path) -> Path:
    """Generate output filename for inference image.

    Args:
        input_path (str | Path): Path to input image
        output_path (str | Path): Path to save output (file or directory)

    Returns:
        Path: Generated output filename

    Raises:
        ValueError: If input_path is not a file

    Examples:
        >>> generate_output_image_filename("input.jpg", "output.jpg")
        PosixPath('output.jpg')  # or output_1.jpg if exists

        >>> generate_output_image_filename("dir/input.jpg", "outdir")
        PosixPath('outdir/dir/input.jpg')
    """
    input_path = validate_path(input_path)
    output_path = validate_path(output_path, should_exist=False)

    if not input_path.is_file():
        msg = "input_path is expected to be a file"
        raise ValueError(msg)

    if output_path.is_dir():
        output_image_filename = output_path / input_path.parent.name / input_path.name
    elif output_path.is_file() and output_path.exists():
        msg = f"{output_path} already exists. Renaming to avoid overwriting."
        logger.warning(msg)
        output_image_filename = duplicate_filename(output_path)
    else:
        output_image_filename = output_path

    output_image_filename.parent.mkdir(parents=True, exist_ok=True)

    return output_image_filename


def get_image_height_and_width(image_size: int | Sequence[int]) -> tuple[int, int]:
    """Get height and width from image size parameter.

    Args:
        image_size (int | Sequence[int]): Single int for square, or (H,W) sequence

    Returns:
        tuple[int, int]: Image height and width

    Raises:
        TypeError: If image_size is not int or sequence of ints

    Examples:
        >>> get_image_height_and_width(256)
        (256, 256)

        >>> get_image_height_and_width((480, 640))
        (480, 640)

        >>> get_image_height_and_width(256.0)
        Traceback (most recent call last):
            ...
        TypeError: ``image_size`` could be either int or tuple[int, int]
    """
    if isinstance(image_size, int):
        height_and_width = (image_size, image_size)
    elif isinstance(image_size, Sequence):
        height_and_width = int(image_size[0]), int(image_size[1])
    else:
        msg = "``image_size`` could be either int or tuple[int, int]"
        raise TypeError(msg)

    return height_and_width


def read_image(path: str | Path, as_tensor: bool = False) -> torch.Tensor | np.ndarray:
    """Read RGB image from disk.

    Args:
        path (str | Path): Path to image file
        as_tensor (bool): If ``True``, return torch.Tensor. Defaults to ``False``

    Returns:
        torch.Tensor | np.ndarray: Image as tensor or array, normalized to [0,1]

    Examples:
        >>> image = read_image("image.jpg")
        >>> type(image)
        <class 'numpy.ndarray'>

        >>> image = read_image("image.jpg", as_tensor=True)
        >>> type(image)
        <class 'torch.Tensor'>
    """
    image = Image.open(path).convert("RGB")
    return to_dtype(to_image(image), torch.float32, scale=True) if as_tensor else np.array(image) / 255.0


def read_mask(path: str | Path, as_tensor: bool = False) -> torch.Tensor | np.ndarray:
    """Read grayscale mask from disk.

    Args:
        path (str | Path): Path to mask file
        as_tensor (bool): If ``True``, return torch.Tensor. Defaults to ``False``

    Returns:
        torch.Tensor | np.ndarray: Mask as tensor or array

    Examples:
        >>> mask = read_mask("mask.png")
        >>> type(mask)
        <class 'numpy.ndarray'>

        >>> mask = read_mask("mask.png", as_tensor=True)
        >>> type(mask)
        <class 'torch.Tensor'>
    """
    image = Image.open(path).convert("L")
    return Mask(to_image(image).squeeze() / 255, dtype=torch.uint8) if as_tensor else np.array(image)


def read_depth_image(path: str | Path) -> np.ndarray:
    """Read depth image from TIFF file.

    Args:
        path (str | Path): Path to TIFF depth image

    Returns:
        np.ndarray: Depth image array

    Examples:
        >>> depth = read_depth_image("depth.tiff")
        >>> type(depth)
        <class 'numpy.ndarray'>
    """
    path = path if isinstance(path, str) else str(path)
    return tiff.imread(path)


def pad_nextpow2(batch: torch.Tensor) -> torch.Tensor:
    """Pad images to next power of 2 size.

    Finds largest dimension and pads to square power-of-2 size. Handles odd sizes.

    Args:
        batch (torch.Tensor): Batch of images to pad

    Returns:
        torch.Tensor: Padded image batch

    Examples:
        >>> x = torch.randn(1, 3, 127, 128)
        >>> padded = pad_nextpow2(x)
        >>> padded.shape
        torch.Size([1, 3, 128, 128])
    """
    l_dim = 2 ** math.ceil(math.log(max(*batch.shape[-2:]), 2))
    padding_w = [math.ceil((l_dim - batch.shape[-2]) / 2), math.floor((l_dim - batch.shape[-2]) / 2)]
    padding_h = [math.ceil((l_dim - batch.shape[-1]) / 2), math.floor((l_dim - batch.shape[-1]) / 2)]
    return F.pad(batch, pad=[*padding_h, *padding_w])


def show_image(image: np.ndarray | Figure, title: str = "Image") -> None:
    """Display image in window.

    Args:
        image (np.ndarray | Figure): Image or matplotlib figure to display
        title (str): Window title. Defaults to "Image"

    Examples:
        >>> img = read_image("image.jpg")
        >>> show_image(img, title="My Image")
    """
    if isinstance(image, Figure):
        image = figure_to_array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(filename: Path | str, image: np.ndarray | Figure, root: Path | None = None) -> None:
    """Save image to disk.

    Args:
        filename (Path | str): Output filename
        image (np.ndarray | Figure): Image or matplotlib figure to save
        root (Path | None): Optional root dir to save under. Defaults to None

    Examples:
        >>> img = read_image("input.jpg")
        >>> save_image("output.jpg", img)

        >>> save_image("subdir/output.jpg", img, root=Path("results"))
    """
    if isinstance(image, Figure):
        image = figure_to_array(image)

    file_path = Path(filename)
    # if file_path is absolute, then root is ignored
    # so we remove the top level directory from the path
    if file_path.is_absolute() and root:
        file_path = Path(*file_path.parts[2:])  # OS-AGNOSTIC
    if root:
        file_path = root / file_path

    # Make unique file_path if file already exists
    file_path = duplicate_filename(file_path)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(file_path), image)


def figure_to_array(fig: Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy array.

    Args:
        fig (Figure): Matplotlib figure to convert

    Returns:
        np.ndarray: RGB image array

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> plt.plot([1, 2, 3])
        >>> img = figure_to_array(fig)
        >>> type(img)
        <class 'numpy.ndarray'>
    """
    fig.canvas.draw()
    # convert figure to np.ndarray for saving via visualizer
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
