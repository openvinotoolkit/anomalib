"""Inference Entrypoint script."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace
from pathlib import Path

from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.engine import Engine
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a config file")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True, help="Path to image(s) to infer.")
    parser.add_argument("--output", type=str, required=False, help="Path to save the output image(s).")
    parser.add_argument(
        "--visualization_mode",
        type=str,
        required=False,
        default="simple",
        help="Visualization mode.",
        choices=["full", "simple"],
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )

    return parser


def infer(args: Namespace):
    """Run inference."""
    config = get_configurable_parameters(config_path=args.config)
    config.visualization.show_images = args.show
    config.visualization.mode = args.visualization_mode
    if args.output:  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = args.output
    else:
        config.visualization.save_images = False

    # create model and trainer
    model = get_model(config)
    callbacks = get_callbacks(config)
    engine = Engine(callbacks=callbacks, **config.trainer)

    # get the transforms
    transform_config = (
        config.data.init_args.transform_config.eval if "transform_config" in config.data.init_args else None
    )
    image_size = (config.data.init_args.image_size[0], config.data.init_args.image_size[1])
    center_crop = config.data.init_args.get("center_crop")
    if center_crop is not None:
        center_crop = tuple(center_crop)
    normalization = InputNormalizationMethod(config.data.init_args.normalization)
    transform = get_transforms(
        config=transform_config,
        image_size=image_size,
        center_crop=center_crop,
        normalization=normalization,
    )

    # create the dataset
    image_size = (int(config.data.init_args.image_size[0]), int(config.data.init_args.image_size[1]))
    dataset = InferenceDataset(
        args.input,
        image_size=image_size,
        transform=transform,  # type: ignore
    )
    dataloader = DataLoader(dataset)

    # generate predictions
    engine.predict(model=model, dataloaders=[dataloader], ckpt_path=str(args.weights))


if __name__ == "__main__":
    args = get_parser().parse_args()
    infer(args)
