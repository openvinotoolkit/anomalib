"""Fixtures for the model tests."""

from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

from anomalib.data import AnomalibDataModule, MVTec, UCSDped
from anomalib.engine import Engine
from anomalib.models import AnomalyModule, get_model
from anomalib.utils.types import TaskType


def get_objects(
    model_name: str,
    dataset_path: Path,
    project_path: Path,
) -> tuple[AnomalyModule, AnomalibDataModule, Engine]:
    """Return model, dataset, and engine objects.

    Args:
        model_name (str): Name of the model to train
        dataset_path (Path): Path to the root of dummy dataset
        project_path (Path): path to the temporary project folder

    Returns:
        tuple[AnomalyModule, AnomalibDataModule, Engine]: Returns the created objects for model, dataset,
            and engine
    """
    # select task type
    if model_name in ("rkde", "ai_vad"):
        task_type = TaskType.DETECTION
    elif model_name in ("ganomaly", "dfkde"):
        task_type = TaskType.CLASSIFICATION
    else:
        task_type = TaskType.SEGMENTATION

    # set extra model args
    # TODO(ashwinvaidya17): Fix these Edge cases
    # https://github.com/openvinotoolkit/anomalib/issues/1478

    extra_args = {}
    if model_name == "patchcore":
        extra_args["input_size"] = (256, 256)
    elif model_name in ("rkde", "dfkde"):
        extra_args["n_pca_components"] = 2

    # select dataset
    if model_name == "ai_vad":
        # aivad expects UCSD dataset
        dataset = UCSDped(
            root=dataset_path / "ucsdped",
            category="dummy",
            task=task_type,
            train_batch_size=1,
            eval_batch_size=1,
        )
    else:
        # EfficientAd requires that the batch size be lesser than the number of images in the dataset.
        # This is so that the LR step size is not 0.
        dataset = MVTec(
            root=dataset_path / "mvtec",
            category="dummy",
            task=task_type,
            train_batch_size=1,
            eval_batch_size=1,
        )

    model = get_model(model_name, **extra_args)
    engine = Engine(
        logger=False,
        default_root_dir=project_path,
        max_epochs=1,
        devices=1,
        pixel_metrics=["F1Score", "AUROC"],
        task=task_type,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"{project_path}/{model_name}/dummy/weights",
                monitor=None,
                filename="last",
                save_last=True,
                auto_insert_metric_name=False,
            ),
        ],
        # TODO(ashwinvaidya17): Fix these Edge cases
        # https://github.com/openvinotoolkit/anomalib/issues/1478
        max_steps=70000 if model_name == "efficient_ad" else -1,
    )
    return model, dataset, engine
