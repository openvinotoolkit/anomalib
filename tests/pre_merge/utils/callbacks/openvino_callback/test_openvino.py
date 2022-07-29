import os
import tempfile

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from anomalib.utils.callbacks.openvino import OpenVINOCallback
from tests.helpers.config import get_test_configurable_parameters
from tests.pre_merge.utils.callbacks.openvino_callback.dummy_lightning_model import (
    DummyLightningModule,
    FakeDataModule,
)


def test_openvino_model_callback():
    """Tests if an optimized model is created."""

    config = get_test_configurable_parameters(
        config_path="tests/pre_merge/utils/callbacks/openvino_callback/dummy_config.yml"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        config.project.path = tmp_dir
        model = DummyLightningModule(hparams=config)
        model.callbacks = [
            OpenVINOCallback(input_size=config.model.input_size, dirpath=os.path.join(tmp_dir), filename="model"),
            EarlyStopping(monitor=config.model.metric),
        ]
        datamodule = FakeDataModule()
        trainer = pl.Trainer(
            gpus=1,
            callbacks=model.callbacks,
            logger=False,
            checkpoint_callback=False,
            max_epochs=1,
            val_check_interval=3,
        )
        trainer.fit(model, datamodule=datamodule)

        assert os.path.exists(os.path.join(tmp_dir, "model.bin")), "Failed to generate OpenVINO model"
