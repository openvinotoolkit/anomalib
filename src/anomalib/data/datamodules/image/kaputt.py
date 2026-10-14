# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Kaputt Data Module.

This module provides a PyTorch Lightning DataModule for the Kaputt dataset.

The Kaputt dataset is a large-scale dataset for visual defect detection in
logistics settings. It has more than 230,000 images with greater than 29,000
defective instances, and 48,000 distinct objects.

Example:
    Create a Kaputt datamodule::

        >>> from anomalib.data import Kaputt
        >>> datamodule = Kaputt(
        ...     root="./datasets/kaputt",
        ... )

Notes:
    The Kaputt dataset is hosted as a gated dataset on Hugging Face at
    https://huggingface.co/datasets/amazon/kaputt. You need to accept
    the dataset's Terms of Use on the Hugging Face website before you
    can download it.

    Once access is granted, this datamodule downloads and extracts the
    dataset automatically as long as a Hugging Face token is available,
    either via the ``HF_TOKEN`` environment variable or a cached login
    (``hf auth login``). If no token is available, or access has not been
    granted yet, see ``get_download_instructions`` for manual download
    steps.

    The expected directory structure after download is::

        datasets/kaputt/
        ├── datasets/                         # Parquet metadata files
        │   ├── query-train.parquet
        │   ├── query-validation.parquet
        │   ├── query-test.parquet
        │   ├── reference-train.parquet
        │   ├── reference-validation.parquet
        │   └── reference-test.parquet
        │
        ├── query-image/data/<split>/query-data/image/   # Query images
        ├── query-crop/data/<split>/query-data/crop/     # Cropped regions
        ├── query-mask/data/<split>/query-data/mask/     # Segmentation masks
        │
        ├── reference-image/data/<split>/reference-data/image/
        ├── reference-crop/data/<split>/reference-data/crop/
        └── reference-mask/data/<split>/reference-data/mask/

License:
    The Kaputt dataset is released under the Creative Commons
    Attribution-NonCommercial-NoDerivatives 4.0 International License
    (CC BY-NC-ND 4.0).
    https://creativecommons.org/licenses/by-nc-nd/4.0/

Reference:
    Höfer, S., Henning, D., Amiranashvili, A., Morrison, D., Tzes, M.,
    Posner, I., Matvienko, M., Rennola, A., & Milan, A. (2025).
    Kaputt: A Large-Scale Dataset for Visual Defect Detection.
    In IEEE/CVF International Conference on Computer Vision (ICCV).
"""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import TYPE_CHECKING

from lightning_utilities.core.imports import module_available
from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.kaputt import ImageMode, ImageType, KaputtDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode
from anomalib.data.utils.download import extract
from anomalib.utils.path import resolve_dataset_root

if TYPE_CHECKING or module_available("huggingface_hub"):
    from huggingface_hub import get_token, hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    # Errors that indicate the dataset cannot be downloaded automatically
    # (e.g. access not granted, repo/revision missing, offline, local I/O
    # issues) and should fall back to the manual download instructions.
    HF_DOWNLOAD_ERRORS: tuple[type[Exception], ...] = (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        EntryNotFoundError,
        LocalEntryNotFoundError,
        OSError,
    )
else:
    get_token = None
    hf_hub_download = None
    HF_DOWNLOAD_ERRORS = (Exception,)

logger = logging.getLogger(__name__)

# Hugging Face repository hosting the Kaputt dataset.
HF_REPO_ID = "amazon/kaputt"

# Commit SHA of the Hugging Face repo to download from. Pinning a specific
# revision (instead of e.g. "main") avoids depending on mutable repo content
# and satisfies Bandit's B615 (huggingface_unsafe_download) check.
HF_REVISION = "5305ccf4f661d84f19a83de35c782686d79f1c84"

# Archives under `kaputt-release/` in the Hugging Face repo. Each archive is
# extracted into a subdirectory named after itself (minus the .tar.gz
# extension), relative to the dataset root, e.g. `query-image.tar.gz` ->
# `<root>/query-image/`.
HF_ARCHIVE_NAMES = (
    "datasets.tar.gz",
    "query-image.tar.gz",
    "query-crop.tar.gz",
    "query-mask.tar.gz",
    "reference-image.tar.gz",
    "reference-crop.tar.gz",
    "reference-mask.tar.gz",
)


class Kaputt(AnomalibDataModule):
    """Kaputt Datamodule.

    Args:
        root (Path | str | None): Path to the root of the dataset.
            Defaults to ``"./datasets/kaputt"``.
        category (str | None): Category of the dataset (maps to ``item_material``).
            Defaults to ``"book_other"``. Pass ``None`` to load all categories.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        image_type (ImageType | str): Type of images to use.
            Defaults to ``ImageType.IMAGE``.
        image_mode (ImageMode | str): Controls which image sources are used for
            training. Defaults to ``ImageMode.QUERY_ONLY``.
        train_augmentations (Transform | None): Augmentations to apply to the training images.
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode): Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Method to create validation set.
            Defaults to ``ValSplitMode.FROM_DIR`` since Kaputt has native validation split.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed for reproducibility.
            Defaults to ``None``.

    Example:
        Create Kaputt datamodule with default settings::

            >>> datamodule = Kaputt()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

        Use cropped images instead of full images::

            >>> datamodule = Kaputt(image_type=ImageType.CROP)

        Include reference (defect-free) images in training::

            >>> datamodule = Kaputt(image_mode=ImageMode.QUERY_AND_REFERENCE)

        Use only reference images for training (no query images)::

            >>> datamodule = Kaputt(image_mode=ImageMode.REFERENCE_ONLY)

        Create validation set from test data (instead of using native val split)::

            >>> datamodule = Kaputt(
            ...     val_split_mode=ValSplitMode.FROM_TEST,
            ...     val_split_ratio=0.1
            ... )

    Note:
        The Kaputt dataset is gated on Hugging Face
        (https://huggingface.co/datasets/amazon/kaputt). Once you have
        requested and been granted access, this datamodule downloads and
        extracts the dataset automatically using your Hugging Face token
        (``HF_TOKEN`` env var or a cached ``hf auth login``). If access has
        not been granted or no token is available, ``prepare_data`` raises
        with manual download instructions.
    """

    def __init__(
        self,
        root: Path | str | None = "./datasets/kaputt",
        category: str | None = "book_other",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        image_type: ImageType | str = ImageType.IMAGE,
        image_mode: ImageMode | str = ImageMode.QUERY_ONLY,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_DIR,
        val_split_ratio: float = 0.5,
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
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        root = resolve_dataset_root(root, "kaputt")
        self.root = Path(root)
        self._category = category or ""
        self.image_type = ImageType(image_type)
        self.image_mode = ImageMode(image_mode)

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The Kaputt dataset has native train/validation/test splits, so we
            use them directly when val_split_mode is FROM_DIR.
        """
        self.train_data = KaputtDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
            image_type=self.image_type,
            image_mode=self.image_mode,
        )
        self.test_data = KaputtDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
            image_type=self.image_type,
            image_mode=ImageMode.QUERY_ONLY,
        )

        # Kaputt has a native validation split
        if self.val_split_mode == ValSplitMode.FROM_DIR:
            self.val_data = KaputtDataset(
                split=Split.VAL,
                root=self.root,
                category=self.category,
                image_type=self.image_type,
                image_mode=ImageMode.QUERY_ONLY,
            )

    def prepare_data(self) -> None:
        """Check if the dataset is available, downloading it if possible.

        This method checks if the specified dataset is available in the file
        system. If it is not, and the ``huggingface_hub`` package is
        installed, it attempts to automatically download and extract the
        (gated) dataset from Hugging Face using the token from the
        ``HF_TOKEN`` environment variable or a cached ``hf auth login``
        session. Automatic download only succeeds if the user has already
        requested and been granted access to the dataset on Hugging Face.

        Raises:
            FileNotFoundError: If the dataset is not found at the specified
                path and cannot be downloaded automatically (e.g. access has
                not been granted, no token is available, or
                ``huggingface_hub`` is not installed).

        Example:
            Assume the dataset is available on the file system::

                >>> datamodule = Kaputt(
                ...     root="./datasets/kaputt",
                ... )
                >>> datamodule.prepare_data()

            Directory structure should include::

                datasets/kaputt/
                ├── datasets/
                │   ├── query-train.parquet
                │   ├── query-validation.parquet
                │   └── query-test.parquet
                └── query-image/
                    └── data/
                        ├── train/
                        ├── validation/
                        └── test/
        """
        datasets_dir = self.root / "datasets"
        query_train_parquet = datasets_dir / "query-train.parquet"

        if datasets_dir.is_dir() and query_train_parquet.exists():
            logger.info("Found the Kaputt dataset.")
            return

        if not module_available("huggingface_hub"):
            raise FileNotFoundError(get_download_instructions(self.root))

        if not get_token():
            logger.info(
                "No Hugging Face token found (HF_TOKEN env var or cached "
                "``hf auth login`` session). Skipping automatic download.",
            )
            raise FileNotFoundError(get_download_instructions(self.root))

        logger.info(
            "Kaputt dataset not found at %s. Attempting to download it from %s.",
            self.root,
            f"https://huggingface.co/datasets/{HF_REPO_ID}",
        )
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with TemporaryDirectory(dir=self.root) as scratch_dir:
                for archive_name in HF_ARCHIVE_NAMES:
                    subdir = archive_name.removesuffix(".tar.gz")
                    target_dir = self.root / subdir
                    if target_dir.is_dir() and any(target_dir.iterdir()):
                        continue

                    logger.info("Downloading %s from Hugging Face.", archive_name)
                    downloaded_path = Path(
                        hf_hub_download(
                            repo_id=HF_REPO_ID,
                            repo_type="dataset",
                            revision=HF_REVISION,
                            filename=f"kaputt-release/{archive_name}",
                            local_dir=scratch_dir,
                        ),
                    )
                    target_dir.mkdir(parents=True, exist_ok=True)
                    extract(downloaded_path, target_dir)
        except HF_DOWNLOAD_ERRORS as exc:
            raise FileNotFoundError(get_download_instructions(self.root)) from exc

        if not query_train_parquet.exists():
            raise FileNotFoundError(get_download_instructions(self.root))

        for archive_name in HF_ARCHIVE_NAMES:
            subdir = archive_name.removesuffix(".tar.gz")
            target_dir = self.root / subdir
            if not (target_dir.is_dir() and any(target_dir.iterdir())):
                logger.error("Expected extracted directory %s is missing or empty.", target_dir)
                raise FileNotFoundError(get_download_instructions(self.root))


def get_download_instructions(root_path: Path) -> str:
    """Get download instructions for the Kaputt dataset.

    Args:
        root_path: Path where the dataset should be downloaded.

    Returns:
        str: Formatted download instructions.
    """
    return dedent(f"""
        Kaputt dataset not found in {root_path}

        The Kaputt Defect Dataset (KDD) is a gated dataset hosted on Hugging
        Face. To get access:

        1. Create a Hugging Face account at https://huggingface.co
        2. Visit https://huggingface.co/datasets/{HF_REPO_ID}
        3. Read and accept the dataset's Terms of Use (LICENSE)
        4. Once access is granted, you have two options to download the dataset:

        Option 1: Using the Hugging Face CLI (Recommended)
        --------------------------------------------------
        a. Install the Hugging Face CLI:
           pip install -U huggingface_hub

        b. Login to Hugging Face:
           hf auth login

        c. Download the dataset archives:
           hf download \\
               --repo-type dataset \\
               --local-dir {root_path}/kaputt-release {HF_REPO_ID} \\
               --include="kaputt-release/*.tar.gz"

        d. Extract each archive into its matching subdirectory, e.g.:
           tar -xzf {root_path}/kaputt-release/datasets.tar.gz -C {root_path}/datasets
           tar -xzf {root_path}/kaputt-release/query-image.tar.gz -C {root_path}/query-image
           tar -xzf {root_path}/kaputt-release/query-crop.tar.gz -C {root_path}/query-crop
           tar -xzf {root_path}/kaputt-release/query-mask.tar.gz -C {root_path}/query-mask
           tar -xzf {root_path}/kaputt-release/reference-image.tar.gz -C {root_path}/reference-image
           tar -xzf {root_path}/kaputt-release/reference-crop.tar.gz -C {root_path}/reference-crop
           tar -xzf {root_path}/kaputt-release/reference-mask.tar.gz -C {root_path}/reference-mask

        Option 2: Manual Download
        --------------------------
        a. Visit https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/kaputt-release
        b. Download each ``*.tar.gz`` archive manually
        c. Extract each archive as described in step 4 above

        Expected directory structure:
        {root_path}/
        ├── datasets/                         # Parquet metadata files
        │   ├── query-train.parquet
        │   ├── query-validation.parquet
        │   ├── query-test.parquet
        │   ├── reference-train.parquet
        │   ├── reference-validation.parquet
        │   └── reference-test.parquet
        ├── query-image/data/<split>/query-data/image/   # Query images
        ├── query-crop/data/<split>/query-data/crop/     # Cropped regions
        ├── query-mask/data/<split>/query-data/mask/     # Segmentation masks
        ├── reference-image/data/<split>/reference-data/image/
        ├── reference-crop/data/<split>/reference-data/crop/
        └── reference-mask/data/<split>/reference-data/mask/

        Note: Authenticate first via `hf auth login`, or set the HF_TOKEN
              environment variable with your access token.
              To get your token, visit: https://huggingface.co/settings/tokens

        For more information about the dataset, see:
        - Paper: https://arxiv.org/abs/2510.05903
        - Dataset: https://huggingface.co/datasets/{HF_REPO_ID}
    """).strip()
