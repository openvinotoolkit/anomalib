import logging
from abc import ABC, abstractmethod

from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import trainer
from anomalib.models import AnomalyModule

logger = logging.getLogger(__name__)


class BaseLoop(ABC):
    def __init__(self, trainer: "trainer.AnomalibTrainer", stage: TrainerFn):
        self.trainer = trainer
        self.stage: str = stage.value if stage != TrainerFn.VALIDATING else "validation"

    def run(self) -> list[STEP_OUTPUT] | None:
        outputs = None
        self.setup()

        try:
            outputs = self.run_epoch_loop()
        except Exception as exception:
            outputs = exception
        finally:
            self.teardown()
        if isinstance(outputs, Exception):
            raise outputs

        return outputs

    @abstractmethod
    def run_epoch_loop(self) -> list[STEP_OUTPUT] | None:
        """Iterate over epoch"""
        raise NotImplementedError

    @abstractmethod
    def run_batch_loop(self):
        """Iterate over batch"""
        raise NotImplementedError

    def setup(self):
        # Setup callbacks
        self.trainer.fabric.call("setup", trainer=self.trainer, pl_module=self.model, stage=self.stage)

    def teardown(self):
        self.trainer.fabric.call("teardown", trainer=self.trainer, pl_module=self.model, stage=self.stage)

    @property
    def model(self) -> AnomalyModule:
        """Returns the model without the fabric wrapper"""
        return _unwrap_objects(self.trainer.model)
