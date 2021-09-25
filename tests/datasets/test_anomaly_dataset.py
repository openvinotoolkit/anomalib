"""Test Anomaly Dataset"""

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from anomalib.config.config import update_input_size
from anomalib.datasets.anomaly_dataset import AnomalyDataModule
from tests.helpers.dataset import get_dataset_path
from tests.helpers.detection import BBFromMasks

tasks = ["detection", "classification", "segmentation"]


class DummyModule(pl.LightningModule):
    def __init__(self, task: str) -> None:
        super().__init__()
        self.task = task

    def training_step(self, batch, _) -> None:
        images = batch["image"]

        assert images.shape[0] == 1, "Expected batch to be of size 1"

        if not isinstance(images[0], torch.Tensor):
            raise TypeError(f"image should be of type torch.Tensor. Found {type(images[0])}")

    def validation_step(self, batch, _) -> None:
        image_paths, images, labels = batch["image_path"], batch["image"], batch["label"]

        assert images.shape[0] == 1, "Expected batch to be of size 1"

        if not isinstance(image_paths[0], str):
            raise TypeError(f"Path should be of type str. Found {type(image_paths[0])}")

        if not isinstance(images[0], torch.Tensor):
            raise TypeError(f"image should be of type torch.Tensor. Found {type(images[0])}")

        if self.task == "classification":
            assert labels.size() == torch.Size([1]), "Expected labels to be a single number"

        if self.task == "segmentation":
            masks, mask_paths = batch["mask"], batch["mask_path"]

            for path in mask_paths:
                if not isinstance(path, str):
                    raise TypeError(f"Path should be of type str. Found {type(path)}")

            assert masks.shape[0] == 1, "Expected masks batch to be of size 1"
            assert masks.dim() == 3, f"Expected masks to have 3 dimensions. Found {masks.dim()}"

        if self.task == "detection" and "good" not in batch["image_path"][0]:
            boxes, box_labels = batch["bbox_t"]["boxes"], batch["bbox_t"]["labels"]
            if not isinstance(boxes[0][0], torch.Tensor):
                raise TypeError(f"Expected bounding box element to be of type torch.Tensor. Found {type(boxes[0][0])}")
            if not isinstance(box_labels[0], tuple) and not isinstance(box_labels[0][0], str):
                raise TypeError(
                    f"Expected bounding box labels to be of type Tuple[str,...]."
                    f" Found {type(box_labels[0])}, {type(box_labels[0][0])} "
                )

    def test_step(self, batch, _) -> None:
        self.validation_step(batch, _)

    def configure_optimizers(self):
        return None


@pytest.mark.parametrize("task", ["classification", "segmentation", "detection"])
def test_anomaly_dataset(task):
    """Test anomaly dataset using MVTec dataset"""

    DATASET_URL = "ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz"

    config = OmegaConf.load("tests/datasets/dummy_config.yml")
    config = update_input_size(config)  # convert image_size to a tuple

    with BBFromMasks(root=get_dataset_path()):
        datamodule = AnomalyDataModule(
            root=get_dataset_path(),
            url=DATASET_URL,
            category="leather",
            task=task,
            label_format="pascal_voc",
            train_batch_size=1,
            test_batch_size=1,
            num_workers=0,
            transform_params=config.transform,
        )

        model = DummyModule(task=task)
        trainer = pl.Trainer(logger=False, gpus=0, check_val_every_n_epoch=1, max_epochs=1)
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)
