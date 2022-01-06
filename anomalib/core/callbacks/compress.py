"""Callback that compresses a trained model by first exporting to .onnx format, and then converting to OpenVINO IR."""
import os
from typing import Tuple, cast

from pytorch_lightning import Callback, LightningModule

from anomalib.core.model.anomaly_module import AnomalyModule
from anomalib.deploy.optimize import export_convert


class CompressModelCallback(Callback):
    """Callback to compresses a trained model.

    Model is first exported to ``.onnx`` format, and then converted to OpenVINO IR.

    Args:
        input_size (Tuple[int, int]): Tuple of image height, width
        dirpath (str): Path for model output
        filename (str): Name of output model
    """

    def __init__(self, input_size: Tuple[int, int], dirpath: str, filename: str):
        self.input_size = input_size
        self.dirpath = dirpath
        self.filename = filename

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Call when the train ends.

        Converts the model to ``onnx`` format and then calls OpenVINO's model optimizer to get the
        ``.xml`` and ``.bin`` IR files.
        """
        os.makedirs(self.dirpath, exist_ok=True)
        onnx_path = os.path.join(self.dirpath, self.filename + ".onnx")
        pl_module = cast(AnomalyModule, pl_module)
        export_convert(
            model=pl_module,
            input_size=self.input_size,
            onnx_path=onnx_path,
            export_path=self.dirpath,
        )
