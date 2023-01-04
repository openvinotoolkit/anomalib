import os
import tempfile

import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from anomalib.deploy import ExportFormat
from anomalib.utils.callbacks.export import ExportCallback
from tests.helpers.config import get_test_configurable_parameters
from tests.pre_merge.utils.callbacks.export_callback.dummy_lightning_model import (
    DummyLightningModule,
    FakeDataModule,
)


@pytest.mark.parametrize(
    "export_format",
    [ExportFormat.OPENVINO, ExportFormat.ONNX],
)
def test_export_callback(export_format):
    """Tests if an optimized model is created."""

    config = get_test_configurable_parameters(
        config_path="tests/pre_merge/utils/callbacks/export_callback/dummy_config.yml"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        config.results_dir.path = tmp_dir
        model = DummyLightningModule(hparams=config)
        model.callbacks = [
            ExportCallback(
                input_size=config.data.init_args.image_size,
                dirpath=os.path.join(tmp_dir),
                filename="model",
                export_format=export_format,
            ),
            EarlyStopping(monitor=config.model.init_args.metric),
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

        if export_format == ExportFormat.OPENVINO:
            assert os.path.exists(os.path.join(tmp_dir, "openvino/model.bin")), "Failed to generate OpenVINO model"
        elif export_format == ExportFormat.ONNX:
            assert os.path.exists(os.path.join(tmp_dir, "onnx/model.onnx")), "Failed to generate ONNX model"
        else:
            raise ValueError(f"Unknown format {export_format}. Supported modes: onnx or openvino.")
