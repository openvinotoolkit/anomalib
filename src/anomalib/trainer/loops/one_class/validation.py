from anomalib import trainer
from anomalib.trainer.loops.base import BaseLoop


class ValidationLoop(BaseLoop):
    def __init__(self, trainer: "trainer.AnomalibTrainer"):
        super().__init__(trainer)

    def run_epoch_loop(self):
        pass

    def run_batch_loop(self):
        pass

    def _setup(self):
        pass
