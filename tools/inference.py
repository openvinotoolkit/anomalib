"""Inference Entrypoint script."""

import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    # --model_config_path will be deprecated in 0.2.8 and removed in 0.2.9
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--config", type=Path, required=True, help="Path to a model config file")
    parser.add_argument("--weight_path", type=Path, required=True, help="Path to a model weights")
    parser.add_argument("--image_path", type=Path, required=True, help="Path to an image to infer.")
    parser.add_argument("--save_path", type=Path, required=False, help="Path to save the output image.")

    args = parser.parse_args()
    if args.model_config_path is not None:
        warnings.warn(
            message="--model_config_path will be deprecated in v0.2.8 and removed in v0.2.9. Use --config instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        args.config = args.model_config_path

    return args


def infer():
    """Run inference."""
    args = get_args()
    config = get_configurable_parameters(config_path=args.config)
    config.model["weight_file"] = str(args.weight_path)

    model = get_model(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(callbacks=callbacks, **config.trainer)

    dataset = InferenceDataset(args.image_path, image_size=tuple(config.dataset.image_size))
    dataloader = DataLoader(dataset)
    trainer.predict(model=model, dataloaders=[dataloader])


if __name__ == "__main__":
    infer()
