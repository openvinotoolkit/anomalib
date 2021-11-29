import os
import tempfile

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from anomalib.config import get_configurable_parameters
from anomalib.core.callbacks.compress import CompressModelCallback
from tests.core.callbacks.compress_callback.dummy_lightning_model import (
    DummyLightningModule,
    FakeDataModule,
)


def test_compress_model_callback():
    """Tests if an optimized model is created."""

    config = get_configurable_parameters(model_config_path="tests/core/callbacks/compress_callback/dummy_config.yml")

    with tempfile.TemporaryDirectory() as tmp_dir:
        config.project.path = tmp_dir
        model = DummyLightningModule(hparams=config)
        model.callbacks = [
            CompressModelCallback(
                input_size=config.model.input_size, dirpath=os.path.join(tmp_dir), filename="compressed_model"
            ),
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

        assert os.path.exists(os.path.join(tmp_dir, "compressed_model.bin")), "Failed to generate OpenVINO model"
