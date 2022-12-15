from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.cli.helpers import configure_optimizer
from tests.helpers.dataset import TestDataset, get_dataset_path


def run_train_test(config):
    model = get_model(config)
    datamodule = get_datamodule(config)
    callbacks = get_callbacks(config)
    configure_optimizer(model, config)
    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    results = trainer.test(model=model, datamodule=datamodule)
    return results


@TestDataset(num_train=200, num_test=30, path=get_dataset_path(), seed=42)
def test_normalizer(path=get_dataset_path(), category="shapes"):
    config = get_configurable_parameters(config_path="anomalib/models/padim/config.yaml")
    config.data.init_args.root = path
    config.data.init_args.category = category
    config.post_processing.threshold_method = "ADAPTIVE"
    config.logging.log_images_to = []
    config.metrics.image_metrics = ["F1Score", "AUROC"]

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
    for metric in ["image_AUROC", "image_F1Score"]:
        assert round(results_without_normalization[0][metric], 3) == round(results_with_cdf_normalization[0][metric], 3)
        assert round(results_without_normalization[0][metric], 3) == round(
            results_with_minmax_normalization[0][metric], 3
        )
