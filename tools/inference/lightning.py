"""Inference Entrypoint script."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
    parser.add_argument("--config", type=Path, required=True, help="Path to a model config file")
    parser.add_argument("--weight_path", type=Path, required=True, help="Path to a model weights")
    parser.add_argument("--image_path", type=Path, required=True, help="Path to an image to infer.")
    parser.add_argument(
        "--visualization_mode",
        type=str,
        required=False,
        default="simple",
        help="Visualization mode. 'full' or 'simple'",
        choices=["full", "simple"],
    )
    parser.add_argument(
        "--disable_show_images",
        action="store_true",
        required=False,
        help="Do not show the visualized predictions on the screen.",
    )
    parser.add_argument("--save_path", type=str, required=False, help="Path to save the output images.")

    args = parser.parse_args()
    return args


def infer():
    """Run inference."""
    args = get_args()
    config = get_configurable_parameters(config_path=args.config)
    config.model["weight_file"] = str(args.weight_path)
    config.visualization.show_images = not args.disable_show_images
    config.visualization.mode = args.visualization_mode
    if args.save_path:  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = args.save_path
    else:
        config.visualization.save_images = False

    model = get_model(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(callbacks=callbacks, **config.trainer)

    dataset = InferenceDataset(args.image_path, image_size=tuple(config.dataset.image_size))
    dataloader = DataLoader(dataset)
    trainer.predict(model=model, dataloaders=[dataloader])


if __name__ == "__main__":
    infer()
