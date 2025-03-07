"""Real-IAD Data Module.

This module provides a PyTorch Lightning DataModule for the Real-IAD dataset.

The Real-IAD dataset is a large-scale industrial anomaly detection dataset containing
30 categories of industrial objects with both normal and anomalous samples. Each object
is captured from 5 different camera viewpoints (C1-C5).

Dataset Structure:
    The dataset follows this directory structure:
        Real-IAD/
        ├── realiad_256/      # 256x256 resolution images
        ├── realiad_512/      # 512x512 resolution images
        ├── realiad_1024/     # 1024x1024 resolution images
        └── realiad_jsons/    # JSON metadata files
            ├── realiad_jsons/       # Base metadata
            ├── realiad_jsons_sv/    # Single-view metadata
            └── realiad_jsons_fuiad/ # FUIAD metadata versions

Example:
    Create a Real-IAD datamodule::

        >>> from anomalib.data import RealIAD
        >>> datamodule = RealIAD(
        ...     root="./datasets/Real-IAD",
        ...     category="audiojack",
        ...     resolution="1024"
        ... )

Notes:
    The dataset should be downloaded manually from Hugging Face and placed in the
    appropriate directory. See ``DOWNLOAD_INSTRUCTIONS`` for detailed steps.

License:
    Real-IAD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from textwrap import dedent

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.realiad import CATEGORIES, RESOLUTIONS, RealIADDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode


class RealIAD(AnomalibDataModule):
    """Real-IAD Datamodule.

    Args:
        root (Path | str): Path to root directory containing the dataset.
            Defaults to ``"./datasets/Real-IAD"``.
        category (str): Category of the Real-IAD dataset (e.g. ``"audiojack"`` or
            ``"button_battery"``). Defaults to ``"audiojack"``.
        resolution (str | int): Image resolution to use (e.g. ``"256"``, ``"512"``,
            ``"1024"``, ``"raw"`` or their integer equivalents).
            For example, both "256" and 256 are valid. Defaults to ``256``.
        json_path (str | Path): Path to JSON metadata file, relative to root directory.
            Can use {category} placeholder which will be replaced with the category name.
            Common paths are:
            - "realiad_jsons/realiad_jsons/{category}.json" - Base metadata (multi-view)
            - "realiad_jsons/realiad_jsons_sv/{category}.json" - Single-view metadata
            - "realiad_jsons/realiad_jsons_fuiad_0.4/{category}.json" - FUIAD v0.4 metadata
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply to the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode): Method to create test set.
            Defaults to ``TestSplitMode.NONE``.
        val_split_mode (ValSplitMode): Method to create validation set.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        seed (int | None, optional): Seed for reproducibility.
            Defaults to ``None``.

    Example:
        Create Real-IAD datamodule with default settings::

            >>> datamodule = RealIAD()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

        Change the category and resolution::

            >>> # Using string resolution
            >>> datamodule = RealIAD(
            ...     category="button_battery",
            ...     resolution="512"
            ... )

            >>> # Using integer resolution
            >>> datamodule = RealIAD(
            ...     category="button_battery",
            ...     resolution=1024
            ... )

        Use different JSON metadata files::

            >>> # Base metadata (multi-view)
            >>> datamodule = RealIAD(
            ...     json_path="realiad_jsons/realiad_jsons/{category}.json"
            ... )

            >>> # Single-view metadata
            >>> datamodule = RealIAD(
            ...     json_path="realiad_jsons/realiad_jsons_sv/{category}.json"
            ... )

            >>> # FUIAD v0.4 metadata (filtered subset)
            >>> datamodule = RealIAD(
            ...     json_path="realiad_jsons/realiad_jsons_fuiad_0.4/{category}.json"
            ... )

            >>> # Custom metadata
            >>> datamodule = RealIAD(
            ...     json_path="path/to/custom/metadata.json"
            ... )

        Create validation set from test data::

            >>> datamodule = RealIAD(
            ...     val_split_mode=ValSplitMode.FROM_TEST,
            ...     val_split_ratio=0.1
            ... )

    Notes:
        - The dataset contains both normal (OK) and anomalous (NG) samples
        - Each object is captured from 5 different camera viewpoints (C1-C5)
        - Images are available in multiple resolutions (256x256, 512x512, 1024x1024)
        - JSON metadata files provide additional information and different dataset splits
        - Segmentation masks are provided for anomalous samples
    """

    def __init__(
        self,
        root: Path | str = "./datasets/Real-IAD",
        category: str = "audiojack",
        resolution: str | int = 256,
        json_path: str | Path = "realiad_jsons/realiad_jsons/{category}.json",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.NONE,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            val_split_mode=val_split_mode,
            seed=seed,
        )

        self.root = Path(root)
        self.category = category

        # Convert resolution to string if it's an integer
        if isinstance(resolution, int):
            resolution = str(resolution)

        self.resolution = resolution
        self.json_path = json_path

        # Validate inputs
        if category not in CATEGORIES:
            msg = f"Category {category} not found in Real-IAD dataset. Available categories: {CATEGORIES}"
            raise ValueError(msg)

        if resolution not in RESOLUTIONS:
            msg = f"Resolution {resolution} not found in Real-IAD dataset. Available resolutions: {RESOLUTIONS}"
            raise ValueError(msg)

    def prepare_data(self) -> None:
        """Verify that the dataset is available and provide download instructions.

        This method checks if the dataset exists in the root directory. If not, it provides
        instructions for requesting access and downloading from Hugging Face.

        The Real-IAD dataset is available at:
        https://huggingface.co/datasets/REAL-IAD/Real-IAD

        Note:
            The dataset requires approval from the authors. You need to:
            1. Create a Hugging Face account
            2. Request access to the dataset
            3. Wait for approval
            4. Download and extract to the root directory
        """
        root_path = Path(self.root)
        required_dirs = [root_path / f"realiad_{res}" for res in RESOLUTIONS] + [
            root_path / "realiad_jsons",
            root_path / "realiad_jsons_sv",
            root_path / "realiad_jsons_fuiad_0.0",
            root_path / "realiad_jsons_fuiad_0.1",
            root_path / "realiad_jsons_fuiad_0.2",
            root_path / "realiad_jsons_fuiad_0.4",
        ]

        if not any(d.exists() for d in required_dirs):
            raise RuntimeError(get_download_instructions(root_path))

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting."""
        self.train_data = RealIADDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
            resolution=self.resolution,
            json_path=self.json_path,
        )
        self.test_data = RealIADDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
            resolution=self.resolution,
            json_path=self.json_path,
        )


def get_download_instructions(root_path: Path) -> str:
    """Get download instructions for the Real-IAD dataset.

    Args:
        root_path: Path where the dataset should be downloaded.

    Returns:
        str: Formatted download instructions.
    """
    return dedent(f"""
        Real-IAD dataset not found in {root_path}

        The Real-IAD dataset requires approval from the authors. To get access:

        1. Create a Hugging Face account at https://huggingface.co
        2. Visit https://huggingface.co/datasets/REAL-IAD/Real-IAD
        3. Click "Access Repository" and fill out the form
        4. Wait for approval from the dataset authors
        5. Once approved, you have two options to download the dataset:

        Option 1: Using Hugging Face CLI (Recommended)
        --------------------------------------------
        a. Install the Hugging Face CLI:
           pip install huggingface_hub

        b. Login to Hugging Face:
           huggingface-cli login

        c. Download the dataset:
           huggingface-cli download \
               --repo-type dataset \
               --local-dir {root_path} REAL-IAD/Real-IAD \
               --include="*" \
               --token YOUR_HF_TOKEN

        Option 2: Manual Download
        -----------------------
        a. Visit https://huggingface.co/datasets/REAL-IAD/Real-IAD
        b. Download all files manually
        c. Extract the contents to: {root_path}

        Expected directory structure:
        {root_path}/
        ├── realiad_256/      # 256x256 resolution images
        ├── realiad_512/      # 512x512 resolution images
        ├── realiad_1024/     # 1024x1024 resolution images
        ├── realiad_raw/      # Original resolution images
        └── realiad_jsons/          # Base JSON metadata
            ├── realiad_jsons_sv/       # Single-view JSON metadata
            ├── realiad_jsons_fuiad_0.0/  # FUIAD v0.0 metadata
            ├── realiad_jsons_fuiad_0.1/  # FUIAD v0.1 metadata
            ├── realiad_jsons_fuiad_0.2/  # FUIAD v0.2 metadata
            └── realiad_jsons_fuiad_0.4/  # FUIAD v0.4 metadata

        Note: Replace YOUR_HF_TOKEN with your Hugging Face access token.
              To get your token, visit: https://huggingface.co/settings/tokens

        For more information about the dataset, see:
        - Paper: https://arxiv.org/abs/2401.02749
        - Code: https://github.com/REAL-IAD/REAL-IAD
        - Dataset: https://huggingface.co/datasets/REAL-IAD/Real-IAD
    """)
