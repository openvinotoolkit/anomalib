from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.core.callbacks import get_callbacks
from anomalib.data import get_datamodule
from anomalib.models import get_model


def test_normalizer():
    config = get_configurable_parameters(model_config_path="anomalib/models/padim/config.yaml")
    config.model.threshold.adaptive = True

    # run with normalization
    config.model.normalize_scores = True
    seed_everything(42)
    results_with_normalization = run_train_test(config)

    # run without normalization
    config.model.normalize_scores = False
    seed_everything(42)
    results_without_normalization = run_train_test(config)

    # performance should be the same
    for metric in ["image_AUROC", "image_F1"]:
        assert results_without_normalization[0][metric] == results_with_normalization[0][metric]


def run_train_test(config):
    model = get_model(config)
    datamodule = get_datamodule(config)
    callbacks = get_callbacks(config)
    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    results = trainer.test(model=model, datamodule=datamodule)
    return results
