import shutil
import tempfile
from pathlib import Path

from anomalib.utils.loggers.tensorboard import AnomalibTensorBoardLogger


class DummyLogger(AnomalibTensorBoardLogger):
    def __init__(self):
        self.tempdir = Path(tempfile.mkdtemp())
        super().__init__(name="tensorboard_logs", save_dir=self.tempdir)

    def __del__(self):
        if self.tempdir.exists():
            shutil.rmtree(self.tempdir)
