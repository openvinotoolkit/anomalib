"""Test batch bounding boxes."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory

import numpy as np
import openvino.runtime as ov
import pytest
import torch

from anomalib.deploy.model import ExportModel


def inputs_targets():
    test_image1 = np.zeros((20, 20), np.uint8)
    test_image1[2:3, 2:3] = 255
    test_image1[5:15, 5:15] = 255
    test_image1[18:20, 18:20] = 255
    test_image1 = test_image1.reshape(1, 1, 20, 20)
    test_image1 = np.vstack([test_image1, np.rot90(test_image1, axes=(2, 3))])
    target1 = np.array([[2, 2, 2, 2], [5, 5, 14, 14], [18, 18, 19, 19], [18, 1, 19, 1], [5, 5, 14, 14], [2, 17, 2, 17]])

    test_image2 = np.zeros((20, 20), np.uint8)
    test_image2[2:3, 2:3] = 255
    test_image2[2:3, 5:10] = 255
    test_image2[5:10, 5:10] = 255
    test_image2[5:10, 15:20] = 255
    test_image2[12:15, 3:5] = 255
    test_image2[15:20, 15:20] = 255
    test_image2 = test_image2.reshape(1, 1, 20, 20)
    target2 = np.array([[2, 2, 2, 2], [5, 2, 9, 2], [5, 5, 9, 9], [15, 5, 19, 9], [3, 12, 4, 14], [15, 15, 19, 19]])

    test_image3 = np.zeros((1, 1, 20, 20), np.uint8)  # empty image
    target3 = np.array([])
    return zip([test_image1, test_image2, test_image3], [target1, target2, target3])


class Dummy(torch.nn.Module):
    def forward(self, image):
        image = image.div(255.0)
        return ExportModel.batch_mask_to_boxes(image, num_iterations=200)


@pytest.fixture(scope="module")
def model():
    with TemporaryDirectory() as tmpdir:
        torch.onnx.export(
            Dummy(),
            torch.randint(0, 255, (1, 1, 20, 20)),
            f"{tmpdir}/bb.onnx",
            input_names=["input"],
            dynamic_axes={"input": {0: "batch_size"}},
        )
        core = ov.Core()
        onnx_model = core.read_model(f"{tmpdir}/bb.onnx")
        ov.serialize(onnx_model, f"{tmpdir}/bb.xml")
        openvino_model = core.read_model(f"{tmpdir}/bb.xml")
        compiled_model = core.compile_model(openvino_model, "CPU")
        infer_request = compiled_model.create_infer_request()
        yield infer_request


@pytest.mark.parametrize(["inputs", "targets"], inputs_targets())
def test_batch_bounding_boxes(inputs, targets, model):
    output = model.infer(inputs)
    assert output[0].all() == targets.all()
