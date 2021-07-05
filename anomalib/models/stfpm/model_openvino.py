"""
OpenVINO Inference of STFPM Algorithm
"""

import os
from pathlib import Path

import numpy as np
from openvino.inference_engine import IECore
from sklearn.metrics import roc_auc_score

from anomalib.core.callbacks.timer import TimerCallback
from anomalib.models.base.model import BaseAnomalySegmentationLightning


class STFPMOpenVino(BaseAnomalySegmentationLightning):
    """PyTorch Lightning module for the STFPM algorithm."""

    def __init__(self, hparams):
        super().__init__(hparams)
        ie_core = IECore()
        bin_path = os.path.join(hparams.project.path, hparams.weight_file)
        xml_path = os.path.splitext(bin_path)[0] + ".xml"
        net = ie_core.read_network(xml_path, bin_path)
        net.batch_size = 1
        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))

        self.exec_net = ie_core.load_network(network=net, device_name="CPU")

        self.callbacks = [TimerCallback()]

    @staticmethod
    def configure_optimizers():
        """
        configure_optimizers [summary]

        Returns:
            None: No optimizer is returned
        """
        # this module is only used in test mode, no need to configure optimizers
        return None

    def test_step(self, batch, _):
        """
        test_step [summary]

        Args:
            batch ([type]): [description]
            _ ([type]): [description]

        Returns:
            [type]: [description]
        """
        filenames, images, labels, masks = batch["image_path"], batch["image"], batch["label"], batch["mask"]
        images = images.cpu().numpy()

        anomaly_maps = self.exec_net.infer(inputs={self.input_blob: images})
        anomaly_maps = list(anomaly_maps.values())

        return {
            "filenames": filenames,
            "images": images,
            "true_labels": labels.cpu().numpy(),
            "true_masks": masks.squeeze().cpu().numpy(),
            "anomaly_maps": anomaly_maps,
        }

    def test_epoch_end(self, outputs):
        """
        test_epoch_end [summary]

        Args:
            outputs ([type]): [description]
        """
        self.filenames = [Path(f) for x in outputs for f in x["filenames"]]
        self.images = [x["images"] for x in outputs]

        self.true_masks = np.stack([output["true_masks"] for output in outputs])
        self.anomaly_maps = np.stack([output["anomaly_maps"] for output in outputs])

        self.true_labels = np.stack([output["true_labels"] for output in outputs])
        self.pred_labels = self.anomaly_maps.reshape(self.anomaly_maps.shape[0], -1).max(axis=1)

        self.image_roc_auc = roc_auc_score(self.true_labels, self.pred_labels)
        self.pixel_roc_auc = roc_auc_score(self.true_masks.flatten(), self.anomaly_maps.flatten())

        self.log(name="Image-Level AUC", value=self.image_roc_auc, on_epoch=True, prog_bar=True)
        self.log(name="Pixel-Level AUC", value=self.pixel_roc_auc, on_epoch=True, prog_bar=True)
