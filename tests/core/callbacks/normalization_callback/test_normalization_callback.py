from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.core.callbacks import get_callbacks
from anomalib.data import get_datamodule
from anomalib.models import get_model
from tests.helpers.dataset import get_dataset_path


def run_train_test(config):
    model = get_model(config)
    datamodule = get_datamodule(config)
    callbacks = get_callbacks(config)
    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    results = trainer.test(model=model, datamodule=datamodule)
    return results


def test_normalizer():
    config = get_configurable_parameters(model_config_path="anomalib/models/padim/config.yaml")
    config.dataset.path = get_dataset_path(config.dataset.path)
    config.model.threshold.adaptive = True

    # run without normalization
    config.model.normalization_method = "none"
    seed_everything(42)
    results_without_normalization = run_train_test(config)

    # run with cdf normalization
    config.model.normalization_method = "cdf"
    seed_everything(42)
    results_with_cdf_normalization = run_train_test(config)

    # run without normalization
    config.model.normalization_method = "min_max"
    seed_everything(42)
    results_with_minmax_normalization = run_train_test(config)

    # performance should be the same
    for metric in ["image_AUROC", "image_F1"]:
        assert round(results_without_normalization[0][metric], 3) == round(results_with_cdf_normalization[0][metric], 3)
        assert round(results_without_normalization[0][metric], 3) == round(
            results_with_minmax_normalization[0][metric], 3
        )
