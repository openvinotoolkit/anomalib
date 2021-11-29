from typing import List, Tuple

import numpy as np
from skimage.draw import polygon


def random_square_patch(input_region: List[int], min_width: int = 10) -> List[int]:
    """Gets a random patch in the input region.

    Args:
        input_region (List[int]): Coordinates of the input region. [x1, y1, x2, y2]
        min_width (int): Minimum width of the returned patch.
    Example:
    >>> image = np.zeros((200,200,3))
    >>> x1, y1, x2, y2 = random_square_patch([100,100,200,200])
    >>> patched = image.copy()
    >>> patched[y1:y2, x1:x2, :] = 1
    >>> plt.imshow(patched)

    Returns:
        List[int]: Random square patch [x1, y1, x2, y2]
    """
    x1_i, y1_i, x2_i, y2_i = input_region
    cx, cy = np.random.randint(x1_i, x2_i), np.random.randint(y1_i, y2_i)
    shortest_dim = min(x2_i - x1_i, y2_i - y1_i)
    # make sure that shortest_dim is larger than min_width
    shortest_dim = max(shortest_dim, min_width + 1)
    rand_half_width = np.random.randint(min_width, shortest_dim) // 2
    x1, y1, x2, y2 = cx - rand_half_width, cy - rand_half_width, cx + rand_half_width, cy + rand_half_width

    # border check
    if x1 < 0:
        x1 = 0
        x2 = 2 * rand_half_width
    elif x2 > x2_i:
        x2 = x2_i
        x1 = x2_i - 2 * rand_half_width

    if y1 < 0:
        y1 = 0
        y2 = 2 * rand_half_width
    elif y2 > y2_i:
        y2 = y2_i
        y1 = y2_i - 2 * rand_half_width

    return [x1, y1, x2, y2]


def triangle(input_region: List[int]) -> Tuple[List[int], List[int]]:
    """Get coordinates of points inside a triangle.

    Args:
        input_region (List[int]): Region in which to draw the triangle. [x1, y1, x2, y2]
    Example:
    >>> image = np.full((200,200,3),fill_value=255, dtype=np.uint8)
    >>> patch_region = random_square_patch([100, 100, 200, 200])
    >>> xx, yy = triangle(patch_region)
    >>> patched = image.copy()
    >>> patched[yy, xx, :] = 1
    >>> plt.imshow(patched)
    Returns:
        Tuple[List[int], List[int]]: Array of cols and rows which denote the mask.
    """
    x1_i, y1_i, x2_i, y2_i = input_region

    x1, y1 = x1_i + (x2_i - x1_i) // 2, y1_i
    x2, y2 = x1_i, y2_i
    x3, y3 = x2_i, y2_i
    return polygon([x1, x2, x3], [y1, y2, y3])


def rectangle(input_region: List[int], min_side: int = 10) -> Tuple[List[int], List[int]]:
    """Get coordinates of corners of a rectangle. Only vertical rectangles are
    generated.

    Args:
        input_region (List[int]): Region in which to draw the rectangle. [x1, y1, x2, y2]
        min_side (int, optional): Minimum side of the rectangle. Defaults to 10.
    Example:
    >>> image = np.full((200,200,3),fill_value=255, dtype=np.uint8)
    >>> patch_region = random_square_patch([100, 100, 200, 200])
    >>> x1, y1, x2, y2 = rectangle(patch_region)
    >>> patched = image.copy()
    >>> patched[y1:y2, x1:x2, :] = 1
    >>> plt.imshow(patched)
    Returns:
        Tuple[List[int], List[int]]: Random rectangle region. [x1, y1, x2, y2]
    """
    x1_i, y1, x2_i, y2 = input_region
    shortest_dim = min(x2_i - x1_i, y2 - y1)
    # make sure that shortest_dim is larger than min_side
    shortest_dim = max(shortest_dim, min_side + 1)
    cx = (x2_i - x1_i) // 2
    rand_half_width = np.random.randint(min_side, shortest_dim) // 2
    x1 = cx - rand_half_width
    x2 = cx + rand_half_width

    xs = np.arange(x1, x2, 1)
    ys = np.arange(y1, y2, 1)

    yy, xx = np.meshgrid(ys, xs, sparse=True)

    return xx, yy


def hexagon(input_region: List[int]) -> Tuple[List[int], List[int]]:
    """Get coordinates of points inside a hexagon.

    Args:
        input_region (List[int]): Region in which to draw the hexagon. [x1, y1, x2, y2]
    Example:
    >>> image = np.full((200,200,3),fill_value=255, dtype=np.uint8)
    >>> patch_region = random_square_patch([100, 100, 200, 200])
    >>> xx, yy = hexagon(patch_region)
    >>> patched = image.copy()
    >>> patched[yy, xx, :] = 1
    >>> plt.imshow(patched)
    Returns:
        Tuple[List[int], List[int]]: Array of cols and rows which denote the mask.
    """
    x1_i, y1_i, x2_i, _ = input_region

    cx = (x2_i - x1_i) // 2
    hex_half_side = (x2_i - x1_i) // 4  # assume side of hexagon to be 1/2 of the square size

    x1, y1 = x1_i + hex_half_side, y1_i
    x2, y2 = x1_i + cx + hex_half_side, y1_i
    x3, y3 = x2_i, y1_i + int(1.732 * hex_half_side)  # 2cos(30)
    x4, y4 = x1_i + cx + hex_half_side, y1_i + int(3.4641 * hex_half_side)  # 4 * cos(30)
    x5, y5 = x1_i + hex_half_side, y1_i + int(3.4641 * hex_half_side)  # 4 * cos(30)
    x6, y6 = x1_i, y1_i + int(1.732 * hex_half_side)
    return polygon([x1, x2, x3, x4, x5, x6], [y1, y2, y3, y4, y5, y6])


def star(input_region: List[int]) -> Tuple[List[int], List[int]]:
    """Get coordinates of points inside a star.

    Args:
        input_region (List[int]): Region in which to draw the star. [x1, y1, x2, y2]
    Example:
    >>> image = np.full((200,200,3),fill_value=255, dtype=np.uint8)
    >>> patch_region = random_square_patch([100, 100, 200, 200])
    >>> xx, yy = star(patch_region)
    >>> patched = image.copy()
    >>> patched[yy, xx, :] = 1
    >>> plt.imshow(patched)
    Returns:
        Tuple[List[int], List[int]]: Array of cols and rows which denote the mask.
    """
    x1_i, y1_i, x2_i, y2_i = input_region

    outer_dim = (x2_i - x1_i) // 2
    inner_dim = (x2_i - x1_i) // 4

    cx = x1_i + (x2_i - x1_i) // 2
    cy = y1_i + (y2_i - y1_i) // 2
    x1, y1 = cx + int(outer_dim * np.cos(0.314159)), cy + int(outer_dim * np.sin(0.314159))
    x2, y2 = cx + int(inner_dim * np.cos(0.942478)), cy + int(inner_dim * np.sin(0.942478))

    x3, y3 = cx + int(outer_dim * np.cos(1.5708)), cy + int(outer_dim * np.sin(1.5708))
    x4, y4 = cx + int(inner_dim * np.cos(2.19911)), cy + int(inner_dim * np.sin(2.19911))

    x5, y5 = cx + int(outer_dim * np.cos(2.82743)), cy + int(outer_dim * np.sin(2.82743))
    x6, y6 = cx + int(inner_dim * np.cos(3.45575)), cy + int(inner_dim * np.sin(3.45575))

    x7, y7 = cx + int(outer_dim * np.cos(4.08407)), cy + int(outer_dim * np.sin(4.08407))
    x8, y8 = cx, cy - inner_dim

    x9, y9 = cx + int(outer_dim * np.cos(5.34071)), cy + int(outer_dim * np.sin(5.34071))
    x10, y10 = cx + int(inner_dim * np.cos(5.96903)), cy + int(inner_dim * np.sin(5.96903))
    print([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10])

    return polygon([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10])


def random_shapes(
    input_region: List[int], size: Tuple[int, int], max_shapes: int, shape: str = "rectangle"
) -> np.ndarray:
    """Generate image with random shape.

    Args:
        input_region (List[int]): Coordinates of the input region. [x1, y1, x2, y2]
        size (Tuple[int, int]): Size of the input image
        max_shapes (int): Maximum number of shapes of a certain kind to draw
        shape (str): Name of the shape. Defaults to rectangle
    Returns:
        np.ndarray: Image containing the shape
    """
    shape_fn: Tuple[List[int], List[int]]
    if shape == "rectangle":
        shape_fn = rectangle
    elif shape == "triangle":
        shape_fn = triangle
    elif shape == "hexagon":
        shape_fn = hexagon
    elif shape == "star":
        shape_fn = star
    else:
        raise ValueError(f"Shape function {shape} not supported!")

    shape_image: np.ndarray = np.full((*size, 3), fill_value=255, dtype=np.uint8)
    for _ in range(max_shapes):
        image = np.full((*size, 3), fill_value=255, dtype=np.uint8)
        patch_region = random_square_patch(input_region)
        xx, yy = shape_fn(patch_region)
        # assign random colour
        image[yy, xx, :] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        shape_image = np.minimum(image, shape_image)  # since 255 is max

    return shape_image
