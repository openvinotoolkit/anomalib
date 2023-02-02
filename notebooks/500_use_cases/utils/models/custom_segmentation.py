# SegmentationModel implementation based on
# https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/common/python/models

import cv2
import numpy as np
from os import PathLike
from models import model
from openvino.runtime import PartialShape
from notebook_utils import segmentation_map_to_overlay


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


class SegmentationModel(model.Model):
    def __init__(
        self,
        ie,
        model_path: PathLike,
        colormap: np.ndarray = None,
        resize_shape=None,
        sigmoid=False,
        argmax=False,
        rgb=False,
        rotate_and_flip=False
    ):
        """
        Segmentation Model for use with Async Pipeline.

        :param model_path: path to IR model .xml file
        :param colormap: array of shape (num_classes, 3) where colormap[i] contains the RGB color
            values for class i. Optional for binary segmentation, required for multiclass
        :param resize_shape: if specified, reshape the model to this shape
        :param sigmoid: if True, apply sigmoid to model result
        :param argmax: if True, apply argmax to model result
        :param rgb: set to True if the model expects RGB images as input
        """
        super().__init__(ie, model_path)

        self.sigmoid = sigmoid
        self.argmax = argmax
        self.rgb = rgb
        self.rotate_and_flip = rotate_and_flip

        self.net = ie.read_model(model_path)
        self.output_layer = self.net.output(0)
        self.input_layer = self.net.input(0)
        if resize_shape is not None:
            self.net.reshape({self.input_layer: PartialShape(resize_shape)})
        self.image_height, self.image_width = self.input_layer.shape[2], self.input_layer.shape[3]

        if colormap is None and self.output_layer.shape[1] == 1:
            self.colormap = np.array([[0, 0, 0], [0, 0, 255]])
        else:
            self.colormap = colormap
        if self.colormap is None:
            raise ValueError("Please provide a colormap for multiclass segmentation")

    def preprocess(self, inputs):
        """
        Resize the image to network input dimensions and transpose to
        network input shape with N,C,H,W layout.

        Images are expected to have dtype np.uint8 and shape (H,W,3) or (H,W)
        """
        meta = {}
        image = inputs[self.input_layer]
        meta["frame"] = image
        if image.shape[:2] != (self.image_height, self.image_width):
            image = cv2.resize(image, (self.image_width, self.image_height))
        if len(image.shape) == 3:
            input_image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        else:
            input_image = np.expand_dims(np.expand_dims(image, 0), 0)
        return {self.input_layer.any_name: input_image}, meta

    def postprocess(self, outputs, preprocess_meta, to_rgb=False):
        """
        Convert raw network results into a segmentation map with overlay. Returns
        a BGR image for further processing with OpenCV. 
        """
        alpha = 0.4

        if preprocess_meta["frame"].shape[-1] == 3:
            bgr_frame = preprocess_meta["frame"]
            if self.rgb:
                # reverse color channels to convert to BGR
                bgr_frame = bgr_frame[:, :, (2, 1, 0)]
        else:
            # Create BGR image by repeating channels in one-channel image
            bgr_frame = np.repeat(np.expand_dims(preprocess_meta["frame"], -1), 3, 2)
        res = outputs[self.output_layer.any_name].squeeze()

        result_mask_ir = sigmoid(res) if self.sigmoid else res

        if self.argmax:
            result_mask_ir = np.argmax(res, axis=0).astype(np.uint8)
        else:
            result_mask_ir = result_mask_ir.round().astype(np.uint8)
        overlay = segmentation_map_to_overlay(
            bgr_frame, result_mask_ir, alpha, colormap=self.colormap
        )
        if self.rotate_and_flip:
            overlay = cv2.flip(cv2.rotate(overlay, rotateCode=cv2.ROTATE_90_CLOCKWISE), flipCode=1)
        return overlay
