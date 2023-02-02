""""Test bounding box conversion utilities"""

import pytest
import torch
from torch import Tensor

from anomalib.data.utils.boxes import (
    boxes_to_anomaly_maps,
    boxes_to_masks,
    masks_to_boxes,
)


@pytest.fixture
def input_masks():
    masks = []
    masks.append(  # normal and tiny shapes
        Tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    masks.append(  # shapes at edge of image
        Tensor(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            ]
        )
    )
    masks.append(  # diagonally touching shapes
        Tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    masks.append(torch.zeros((10, 10)))  # empty mask
    return torch.stack(masks)


@pytest.fixture
def input_maps():
    masks = []
    masks.append(  # normal and tiny shapes
        Tensor(
            [
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [1, 1, 2, 0, 0, 1, 2, 2, 1, 0],
                [1, 2, 4, 0, 0, 2, 3, 4, 2, 0],
                [1, 2, 3, 0, 3, 4, 6, 5, 3, 0],
                [1, 1, 2, 3, 4, 4, 5, 3, 1, 0],
                [0, 1, 1, 2, 3, 4, 4, 3, 1, 1],
                [0, 1, 2, 2, 2, 3, 3, 2, 1, 0],
                [1, 2, 3, 3, 3, 2, 2, 1, 0, 0],
                [2, 3, 5, 4, 2, 1, 1, 0, 0, 0],
                [1, 2, 3, 3, 2, 1, 0, 0, 0, 0],
            ]
        )
    )
    masks.append(  # shapes at edge of image
        Tensor(
            [
                [0.4, 0.2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.3, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 99999],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    masks.append(  # diagonally touching shapes
        Tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    masks.append(torch.zeros((10, 10)))  # empty mask
    return torch.stack(masks)


@pytest.fixture
def target_scores():
    return [
        Tensor([4, 6, 5]),
        Tensor([0.4, 99999]),
        Tensor([1]),
        Tensor(),
    ]


@pytest.fixture
def target_boxes():
    boxes = []
    boxes.append(Tensor([[2, 2, 2, 2], [4, 2, 7, 5], [2, 8, 3, 8]]))
    boxes.append(Tensor([[0, 0, 1, 1], [8, 8, 9, 0]]))
    boxes.append(Tensor([[1, 4, 3, 7], [4, 2, 5, 3]]))
    boxes.append(torch.empty((0, 4)))
    return boxes


@pytest.fixture
def input_boxes():
    boxes = []
    boxes.append(Tensor([[2, 3, 4, 5], [6, 7, 6, 7]]))  # normal box and tiny box
    boxes.append(Tensor([[0, 0, 1, 1], [8, 8, 9, 9]]))  # boxes at edge of image
    boxes.append(Tensor([[1, 5, 7, 6], [4, 2, 5, 8]]))  # overlapping boxes
    boxes.append(Tensor([[4, 2, 5, 8], [1, 5, 7, 6]]))  # overlapping boxes swapped
    boxes.append(torch.empty((0, 4)))  # no boxes
    return boxes


@pytest.fixture
def input_scores():
    scores = []
    scores.append(Tensor([0.3, 0.5]))
    scores.append(Tensor([6, 0.001]))
    scores.append(Tensor([4, 5]))
    scores.append(Tensor([5, 4]))
    return scores


@pytest.fixture
def target_masks():
    masks = []
    masks.append(
        Tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    masks.append(
        Tensor(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            ]
        )
    )
    masks.append(
        Tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    masks.append(
        Tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    masks.append(torch.zeros((10, 10)))
    return torch.stack(masks)


@pytest.fixture
def target_anomaly_maps():
    maps = []
    maps.append(
        Tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0],
                [0, 0, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0],
                [0, 0, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    maps.append(
        Tensor(
            [
                [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.001],
                [0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.001],
            ]
        )
    )
    maps.append(
        Tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                [0, 4, 4, 4, 5, 5, 4, 4, 0, 0],
                [0, 4, 4, 4, 5, 5, 4, 4, 0, 0],
                [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    maps.append(
        Tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                [0, 4, 4, 4, 5, 5, 4, 4, 0, 0],
                [0, 4, 4, 4, 5, 5, 4, 4, 0, 0],
                [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    maps.append(torch.zeros((10, 10)))
    return torch.stack(maps)


class TestMasksToBoxes:
    def test_output(self, input_masks, target_boxes):
        out_boxes, _ = masks_to_boxes(input_masks)
        assert [out_box == target_box for out_box, target_box in zip(out_boxes, target_boxes)]

    @pytest.mark.parametrize(
        "masks",
        (
            Tensor([1, 0] * 5).repeat(10, 1),  # (H, W)
            Tensor([1, 0] * 5).repeat(1, 10, 1),  # (1, H, W)
            Tensor([1, 0] * 5).repeat(32, 10, 1),  # (B, H, W)
            Tensor([1, 0] * 5).repeat(1, 1, 10, 1),  # (1, 1, H, W)
            Tensor([1, 0] * 5).repeat(32, 1, 10, 1),
        ),
    )  # (B, 1, H, W)
    def test_input_shapes(self, masks):
        out_boxes, _ = masks_to_boxes(masks)
        target_length = 1 if masks.dim() == 2 else masks.shape[0]
        assert len(out_boxes) == target_length
        assert out_boxes[0].shape == torch.Size((5, 4))

    def test_box_scores(self, input_masks, input_maps, target_scores):
        _, out_scores = masks_to_boxes(input_masks, input_maps)
        assert all(torch.all(out == target) for out, target in zip(out_scores, target_scores))


class TestBoxesToMasks:
    def test_output(self, input_boxes, target_masks):
        out_masks = boxes_to_masks(input_boxes, image_size=target_masks.shape[-2:])
        assert torch.all(target_masks == out_masks)


class TestBoxesToAnomalyMaps:
    def test_output(self, input_boxes, input_scores, target_anomaly_maps):
        out_maps = boxes_to_anomaly_maps(input_boxes, input_scores, image_size=target_anomaly_maps.shape[-2:])
        assert torch.all(out_maps == target_anomaly_maps)
