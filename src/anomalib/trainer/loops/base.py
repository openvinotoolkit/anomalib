from abc import ABC, abstractmethod

from anomalib import trainer


class BaseLoop(ABC):
    def __init__(self, trainer: "trainer.AnomalibTrainer"):
        self.trainer = trainer

    def run(self, *args, **kwargs):
        self._setup()
        self.run_epoch_loop(*args, **kwargs)

    @abstractmethod
    def run_epoch_loop(self):
        """Iterate over epoch"""
        raise NotImplementedError

    @abstractmethod
    def run_batch_loop(self):
        """Iterate over batch"""
        raise NotImplementedError

    @abstractmethod
    def _setup(self):
        raise NotImplementedError
