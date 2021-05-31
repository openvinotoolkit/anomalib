from typing import Any, Dict
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule
import nncf
from nncf import NNCFConfig, create_compressed_model, load_state, register_default_init_args
from anomalib.models.stfpm.nncf_utils import InitLoader, criterion_fn


class NNCFCallback(Callback):

    def __init__(self, nncf_config_path):
        self.nncf_config = NNCFConfig.from_json(nncf_config_path)

    def on_before_accelerator_backend_setup(self, trainer, pl_module: LightningModule) -> None:
        """Called before accelerator is being setup"""
        init_loader = InitLoader(pl_module.train_loader)
        nncf_config = register_default_init_args(self.nncf_config, init_loader, pl_module.model.loss,
                                                 criterion_fn=criterion_fn)
        self.comp_ctrl, pl_module.model = create_compressed_model(pl_module.model, nncf_config)
        self.compression_scheduler = self.comp_ctrl.scheduler
        # pass

    def setup(self, trainer, pl_module: LightningModule, stage: str) -> None:
        """Called when fit or test begins"""
        init_loader = InitLoader(pl_module.train_loader)
        nncf_config = register_default_init_args(self.nncf_config, init_loader, pl_module.model.loss,
                                                 criterion_fn=criterion_fn)
        self.comp_ctrl, pl_module.model = create_compressed_model(pl_module.model, nncf_config)
        self.compression_scheduler = self.comp_ctrl.scheduler
        # pass

    def teardown(self, trainer, pl_module: LightningModule, stage: str) -> None:
        """Called when fit or test ends"""
        pass

    def on_init_start(self, trainer) -> None:
        """Called when the trainer initialization begins, model has not yet been set."""
        pass

    def on_init_end(self, trainer) -> None:
        """Called when the trainer initialization ends, model has not yet been set."""
        pass

    def on_fit_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when fit begins"""
        init_loader = InitLoader(pl_module.train_loader)
        nncf_config = register_default_init_args(self.nncf_config, init_loader, pl_module.model.loss,
                                                 criterion_fn=criterion_fn)
        self.comp_ctrl, pl_module.model = create_compressed_model(pl_module.model, nncf_config)
        self.compression_scheduler = self.comp_ctrl.scheduler

    def on_fit_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when fit ends"""
        pass

    def on_sanity_check_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the validation sanity check starts."""
        pass

    def on_sanity_check_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the validation sanity check ends."""
        pass

    def on_train_batch_start(
        self, trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the train batch begins."""
        self.compression_scheduler.step()
        trainer.model.loss_val = self.comp_ctrl.loss()

    def on_train_batch_end(
        self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the train batch ends."""
        pass

    def on_train_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the train epoch begins."""
        pass

    def on_train_epoch_end(self, trainer, pl_module: LightningModule, outputs: Any) -> None:
        """Called when the train epoch ends."""
        pass

    def on_validation_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the val epoch begins."""
        pass

    def on_validation_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the val epoch ends."""
        pass

    def on_test_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the test epoch begins."""
        pass

    def on_test_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the test epoch ends."""
        pass

    def on_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when either of train/val/test epoch begins."""
        pass

    def on_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when either of train/val/test epoch ends."""
        pass

    def on_batch_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the training batch begins."""
        pass

    def on_validation_batch_start(
        self, trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the validation batch begins."""
        pass

    def on_validation_batch_end(
        self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the validation batch ends."""
        pass

    def on_test_batch_start(
        self, trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the test batch begins."""
        pass

    def on_test_batch_end(
        self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the test batch ends."""
        pass

    def on_batch_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the training batch ends."""
        pass

    def on_train_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the train begins."""
        pass

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the train ends."""
        self.comp_ctrl.export_model("compressed_model.onnx")

    def on_pretrain_routine_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the pretrain routine begins."""
        pass

    def on_pretrain_routine_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the pretrain routine ends."""
        pass

    def on_validation_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the validation loop ends."""
        pass

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the test begins."""
        pass

    def on_test_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the test ends."""
        pass

    def on_keyboard_interrupt(self, trainer, pl_module: LightningModule) -> None:
        """Called when the training is interrupted by ``KeyboardInterrupt``."""
        pass

    def on_save_checkpoint(self, trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> dict:
        """
        Called when saving a model checkpoint, use to persist state.

        Args:
            trainer: the current Trainer instance.
            pl_module: the current LightningModule instance.
            checkpoint: the checkpoint dictionary that will be saved.

        Returns:
            The callback state.
        """
        pass

    def on_load_checkpoint(self, callback_state: Dict[str, Any]) -> None:
        """Called when loading a model checkpoint, use to reload state.

        Args:
            callback_state: the callback state returned by ``on_save_checkpoint``.
        """
        pass

    def on_after_backward(self, trainer, pl_module: LightningModule) -> None:
        """Called after ``loss.backward()`` and before optimizers do anything."""
        pass

    def on_before_zero_grad(self, trainer, pl_module: LightningModule, optimizer) -> None:
        """Called after ``optimizer.step()`` and before ``optimizer.zero_grad()``."""
        pass
