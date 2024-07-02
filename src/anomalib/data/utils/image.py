"""Image Utils."""

# Copyright (C) 2022-2024 Intel Corporation
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
    """Check if the filename is an image file.

    Args:
        filename (str | Path): Filename to check.

    Returns:
        bool: True if the filename is an image file.

    Examples:
        >>> is_image_file("000.png")
        True

        >>> is_image_file("002.JPEG")
        True

        >>> is_image_file("009.tiff")
        True

        >>> is_image_file("002.avi")
        False
    """
    filename = Path(filename)
    return filename.suffix.lower() in IMG_EXTENSIONS


def get_image_filename(filename: str | Path) -> Path:
    """Get image filename.

    Args:
        filename (str | Path): Filename to check.

    Returns:
        Path: Image filename.

    Examples:
        Assume that we have the following files in the directory:

        .. code-block:: bash

            $ ls
            000.png  001.jpg  002.JPEG  003.tiff  004.png  005.txt

        >>> get_image_filename("000.png")
        PosixPath('000.png')

        >>> get_image_filename("001.jpg")
        PosixPath('001.jpg')

        >>> get_image_filename("009.tiff")
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in get_image_filename
        FileNotFoundError: File not found: 009.tiff

        >>> get_image_filename("005.txt")
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in get_image_filename
        ValueError: ``filename`` is not an image file. 005.txt
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
    """Get image filenames from directory.

    Args:
        path (str | Path): Path to image directory.

    Raises:
        ValueError: When ``path`` is not a directory.

    Returns:
        list[Path]: Image filenames.

    Examples:
        Assume that we have the following files in the directory:
        $ ls
        000.png  001.jpg  002.JPEG  003.tiff  004.png  005.png

        >>> get_image_filenames_from_dir(".")
        [PosixPath('000.png'), PosixPath('001.jpg'), PosixPath('002.JPEG'),
        PosixPath('003.tiff'), PosixPath('004.png'), PosixPath('005.png')]

        >>> get_image_filenames_from_dir("009.tiff")
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in get_image_filenames_from_dir
        ValueError: ``path`` is not a directory: 009.tiff
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
    """Get image filenames.

    Args:
        path (str | Path): Path to image or image-folder.
        base_dir (Path): Base directory to restrict file access.

    Returns:
        list[Path]: List of image filenames.

    Examples:
        Assume that we have the following files in the directory:

        .. code-block:: bash

            $ tree images
            images
            ├── bad
            │   ├── 003.png
            │   └── 004.jpg
            └── good
                ├── 000.png
                └── 001.tiff

        We can get the image filenames with various ways:

        >>> get_image_filenames("images/bad/003.png")
        PosixPath('/home/sakcay/Projects/anomalib/images/bad/003.png')]

        It is possible to recursively get the image filenames from a directory:

        >>> get_image_filenames("images")
        [PosixPath('/home/sakcay/Projects/anomalib/images/bad/003.png'),
        PosixPath('/home/sakcay/Projects/anomalib/images/bad/004.jpg'),
        PosixPath('/home/sakcay/Projects/anomalib/images/good/001.tiff'),
        PosixPath('/home/sakcay/Projects/anomalib/images/good/000.png')]

        If we want to restrict the file access to a specific directory,
        we can use ``base_dir`` argument.

        >>> get_image_filenames("images", base_dir="images/bad")
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in get_image_filenames
        ValueError: Access denied: Path is outside the allowed directory.
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
    """Check and duplicate filename.

    This function checks the path and adds a suffix if it already exists on the file system.

    Args:
        path (str | Path): Input Path

    Examples:
        >>> path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
        >>> path.exists()
        True

        If we pass this to ``duplicate_filename`` function we would get the following:
        >>> duplicate_filename(path)
        PosixPath('datasets/MVTec/bottle/test/broken_large/000_1.png')

    Returns:
        Path: Duplicated output path.
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
    """Generate an output filename to save the inference image.

    This function generates an output filaname by checking the input and output filenames. Input path is
    the input to infer, and output path is the path to save the output predictions specified by the user.

    The function expects ``input_path`` to always be a file, not a directory. ``output_path`` could be a
    filename or directory. If it is a filename, the function checks if the specified filename exists on
    the file system. If yes, the function calls ``duplicate_filename`` to duplicate the filename to avoid
    overwriting the existing file. If ``output_path`` is a directory, this function adds the parent and
    filenames of ``input_path`` to ``output_path``.

    Args:
        input_path (str | Path): Path to the input image to infer.
        output_path (str | Path): Path to output to save the predictions.
            Could be a filename or a directory.

    Examples:
        >>> input_path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
        >>> output_path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
        >>> generate_output_image_filename(input_path, output_path)
        PosixPath('datasets/MVTec/bottle/test/broken_large/000_1.png')

        >>> input_path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
        >>> output_path = Path("results/images")
        >>> generate_output_image_filename(input_path, output_path)
        PosixPath('results/images/broken_large/000.png')

    Raises:
        ValueError: When the ``input_path`` is not a file.

    Returns:
        Path: The output filename to save the output predictions from the inferencer.
    """
    input_path = validate_path(input_path)
    output_path = validate_path(output_path, should_exist=False)

    # Input validation: Check if input_path is a valid directory or file
    if input_path.is_file() is False:
        msg = "input_path is expected to be a file to generate a proper output filename."
        raise ValueError(msg)

    # If the output is a directory, then add parent directory name
    # and filename to the path. This is to ensure we do not overwrite
    # images and organize based on the categories.
    if output_path.is_dir():
        output_image_filename = output_path / input_path.parent.name / input_path.name
    elif output_path.is_file() and output_path.exists():
        msg = f"{output_path} already exists. Renaming the file to avoid overwriting."
        logger.warning(msg)
        output_image_filename = duplicate_filename(output_path)
    else:
        output_image_filename = output_path

    output_image_filename.parent.mkdir(parents=True, exist_ok=True)

    return output_image_filename


def get_image_height_and_width(image_size: int | Sequence[int]) -> tuple[int, int]:
    """Get image height and width from ``image_size`` variable.

    Args:
        image_size (int | Sequence[int] | None, optional): Input image size.

    Raises:
        ValueError: Image size not None, int or Sequence of values.

    Examples:
        >>> get_image_height_and_width(image_size=256)
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256))
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256, 3))
        (256, 256)

        >>> get_image_height_and_width(image_size=256.)
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in get_image_height_and_width
        ValueError: ``image_size`` could be either int or tuple[int, int]

    Returns:
        tuple[int | None, int | None]: A tuple containing image height and width values.
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
    """Read image from disk in RGB format.

    Args:
        path (str, Path): path to the image file
        as_tensor (bool, optional): If True, returns the image as a tensor. Defaults to False.

    Example:
        >>> image = read_image("test_image.jpg")
        >>> type(image)
        <class 'numpy.ndarray'>
        >>>
        >>> image = read_image("test_image.jpg", as_tensor=True)
        >>> type(image)
        <class 'torch.Tensor'>

    Returns:
        image as numpy array
    """
    image = Image.open(path).convert("RGB")
    return to_dtype(to_image(image), torch.float32, scale=True) if as_tensor else np.array(image) / 255.0


def read_mask(path: str | Path, as_tensor: bool = False) -> torch.Tensor | np.ndarray:
    """Read mask from disk.

    Args:
        path (str, Path): path to the mask file
        as_tensor (bool, optional): If True, returns the mask as a tensor. Defaults to False.

    Example:
        >>> mask = read_mask("test_mask.png")
        >>> type(mask)
        <class 'numpy.ndarray'>
        >>>
        >>> mask = read_mask("test_mask.png", as_tensor=True)
        >>> type(mask)
        <class 'torch.Tensor'>
    """
    image = Image.open(path).convert("L")
    return Mask(to_image(image).squeeze() / 255, dtype=torch.uint8) if as_tensor else np.array(image)


def read_depth_image(path: str | Path) -> np.ndarray:
    """Read tiff depth image from disk.

    Args:
        path (str, Path): path to the image file

    Example:
        >>> image = read_depth_image("test_image.tiff")

    Returns:
        image as numpy array
    """
    path = path if isinstance(path, str) else str(path)
    return tiff.imread(path)


def pad_nextpow2(batch: torch.Tensor) -> torch.Tensor:
    """Compute required padding from input size and return padded images.

    Finds the largest dimension and computes a square image of dimensions that are of the power of 2.
    In case the image dimension is odd, it returns the image with an extra padding on one side.

    Args:
        batch (torch.Tensor): Input images

    Returns:
        batch: Padded batch
    """
    # find the largest dimension
    l_dim = 2 ** math.ceil(math.log(max(*batch.shape[-2:]), 2))
    padding_w = [math.ceil((l_dim - batch.shape[-2]) / 2), math.floor((l_dim - batch.shape[-2]) / 2)]
    padding_h = [math.ceil((l_dim - batch.shape[-1]) / 2), math.floor((l_dim - batch.shape[-1]) / 2)]
    return F.pad(batch, pad=[*padding_h, *padding_w])


def show_image(image: np.ndarray | Figure, title: str = "Image") -> None:
    """Show an image on the screen.

    Args:
        image (np.ndarray | Figure): Image that will be shown in the window.
        title (str, optional): Title that will be given to that window. Defaults to "Image".
    """
    if isinstance(image, Figure):
        image = figure_to_array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(filename: Path | str, image: np.ndarray | Figure, root: Path | None = None) -> None:
    """Save an image to the file system.

    Args:
        filename (Path | str): Path or filename to which the image will be saved.
        image (np.ndarray | Figure): Image that will be saved to the file system.
        root (Path, optional): Root directory to save the image. If provided, the top level directory of an absolute
            filename will be overwritten. Defaults to None.
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
    """Convert a matplotlib figure to a numpy array.

    Args:
        fig (Figure): Matplotlib figure.

    Returns:
        np.ndarray: Numpy array containing the image.
    """
    fig.canvas.draw()
    # convert figure to np.ndarray for saving via visualizer
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
