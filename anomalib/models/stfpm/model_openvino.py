import os
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
from openvino.inference_engine import IECore
from sklearn.metrics import roc_auc_score

from anomalib.core.callbacks.timer import TimerCallback


class AnomalyMapGenerator:
    def __init__(self, batch_size: int = 1, image_size: int = 256, alpha: float = 0.4, gamma: int = 0):
        super(AnomalyMapGenerator, self).__init__()
        self.image_size = image_size
        self.batch_size = batch_size

        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.gamma = gamma

    def compute_layer_map(self, teacher_features, student_features):
        norm_teacher_features = teacher_features / np.linalg.norm(teacher_features, axis=1)
        norm_student_features = student_features / np.linalg.norm(student_features, axis=1)

        layer_map = 0.5 * np.linalg.norm(norm_teacher_features - norm_student_features, ord=2, axis=-3, keepdims=True) ** 2
        layer_map = cv2.resize(layer_map[0, 0, ...], (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return layer_map

    def compute_anomaly_map(self, teacher_features, student_features):
        anomaly_map = np.ones([self.image_size, self.image_size])
        for layer in teacher_features.keys():
            layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer])
            anomaly_map *= layer_map

        return anomaly_map

    @staticmethod
    def compute_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
        anomaly_map = anomaly_map * 255
        anomaly_map = anomaly_map.astype(np.uint8)

        heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        return heatmap

    def apply_heatmap_on_image(self, anomaly_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        heatmap = self.compute_heatmap(anomaly_map)
        heatmap_on_image = cv2.addWeighted(heatmap, self.alpha, image, self.beta, self.gamma)
        return heatmap_on_image

    def __call__(self, teacher_features, student_features):
        return self.compute_anomaly_map(teacher_features, student_features)


class STFPMOpenVino(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        ie = IECore()
        bin_path = os.path.join(hparams.project.path, hparams.weight_file)
        xml_path = os.path.splitext(bin_path)[0] + ".xml"
        net = ie.read_network(xml_path, bin_path)
        net.batch_size = 1
        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))

        self.exec_net = ie.load_network(network=net, device_name="CPU")
        self.anomaly_map_generator = AnomalyMapGenerator(batch_size=1, image_size=224)

        self.callbacks = [TimerCallback()]

    def configure_optimizers(self):
        # this module is only used in test mode, no need to configure optimizers
        return None

    def test_step(self, batch, batch_idx):
        filenames, images, labels, masks = batch["image_path"], batch["image"], batch["label"], batch["mask"]
        images = images.cpu().numpy()

        res = self.exec_net.infer(inputs={self.input_blob: images})
        num_layers = int(len(res)/2)
        layer_names = [f"layer{num}" for num in range(num_layers)]
        teacher_features = {key:value for key, value in zip(layer_names, list(res.values())[:num_layers])}
        student_features = {key: value for key, value in zip(layer_names, list(res.values())[num_layers:])}

        anomaly_maps = self.anomaly_map_generator(teacher_features, student_features)

        return {
            "filenames": filenames,
            "images": images,
            "true_labels": labels.cpu().numpy(),
            "true_masks": masks.squeeze().cpu().numpy(),
            "anomaly_maps": anomaly_maps,
        }

    def test_epoch_end(self, outputs):

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
