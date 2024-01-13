"""MVTec LOCO AD Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch Lightning
    DataModule for the MVTec LOCO AD dataset. If the dataset is not on the file system,
    the script downloads and extracts the dataset and create PyTorch data objects.

License:
    MVTec LOCO AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

References:
    - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger:
      Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization;
      in: International Journal of Computer Vision (IJCV) 130, 947-969, 2022, DOI: 10.1007/s11263-022-01578-9
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import albumentations as A  # noqa: N812
import cv2
import numpy as np
from pandas import DataFrame

from anomalib import TaskType
from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.utils import (
    DownloadInfo,
    InputNormalizationMethod,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    get_transforms,
)

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = (".png", ".PNG")

DOWNLOAD_INFO = DownloadInfo(
    name="mvtec_loco",
    url="https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701"
    "/mvtec_loco_anomaly_detection.tar.xz",
    checksum="d40f092ac6f88433f609583c4a05f56f",
)

CATEGORIES = (
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
)

GT_MERGED_DIR = "ground_truth_merged"


def _merge_gt_mask(
    root: str | Path,
    extensions: Sequence[str] = IMG_EXTENSIONS,
    gt_merged_dir: str | Path = GT_MERGED_DIR,
) -> None:
    """Merges ground truth masks within specified directories and saves the merged masks.

    Args:
        root (str | Path): Root directory containing the 'ground_truth' folder.
        extensions (Sequence[str]): Allowed file extensions for ground truth masks.
                                    Default is IMG_EXTENSIONS.
        gt_merged_dir (str | Path]): Directory where merged masks will be saved.
                                     Default is GT_MERGED_DIR.

    Returns:
        None

    Example:
        >>> _merge_gt_mask('path/to/breakfast_box/')

        This function reads ground truth masks from the specified directories, merges them into
        a single mask for each corresponding images (e.g. merge 059/000.png and 059/001.png into 059.png),
        and saves the merged masks in the default GT_MERGED_DIR directory.

        Note: The merged masks are saved with the same filename structure as the corresponding anomalous image files.
    """
    root = Path(root)
    gt_mask_paths = {f.parent for f in root.glob("ground_truth/**/*") if f.suffix in extensions}

    for mask_path in gt_mask_paths:
        # Merge each mask inside mask_path into a single mask
        merged_mask = None
        for mask_file in mask_path.glob("*"):
            if mask_file.suffix in extensions:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
                if merged_mask is None:
                    merged_mask = np.zeros_like(mask)
                merged_mask = np.maximum(merged_mask, mask)

        # Binarize masks
        merged_mask = np.minimum(merged_mask, 255)

        # Define the path for the new merged mask
        _, anomaly_dir, image_filename = mask_path.parts[-3:]
        new_mask_path = root / Path(gt_merged_dir) / anomaly_dir / (image_filename + ".png")

        # Create the necessary directories if they do not exist
        new_mask_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the merged mask
        cv2.imwrite(str(new_mask_path), merged_mask)


def make_mvtec_loco_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] = IMG_EXTENSIONS,
    gt_merged_dir: str | Path = GT_MERGED_DIR,
) -> DataFrame:
    """Create MVTec LOCO AD samples by parsing the original MVTec LOCO AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/image_filename/000.png

    where there can be multiple ground-truth masks for the corresponding anomalous images.

    This function first merges the multiple ground-truth-masks by executing _merge_gt_mask(),
    it then creates a dataframe to store the parsed information based on the following format:

    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+
    |   | path          | split | label   | image_path    | mask_path                             | label_index |
    +===+===============+=======+=========+===============+=======================================+=============+
    | 0 | datasets/name | test  | defect  | filename.png  | path/to/merged_masks/filename.png      | 1           |
    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+

    Note: the final image_path is converted to full path by combining it with the path, split, and label columns
    Example, datasets/name/test/defect/filename.png

    Args:
        root (str | Path): Path to dataset
        split (str | Split | None): Dataset split (ie., either train or test).
            Defaults to ``None``.
        extensions (Sequence[str]): List of file extensions to be included in the dataset.
            Defaults to ``None``.
        gt_merged_dir (str | Path]): Directory where merged masks will be saved.
                                             Default is GT_MERGED_DIR.

    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.

    Examples:
        The following example shows how to get test samples from MVTec LOCO AD pushpins category:

        >>> root = Path('./MVTec_LOCO')
        >>> category = 'pushpins'
        >>> path = root / category
        >>> samples = make_mvtec_loco_dataset(path, split='test')
    """
    root = Path(root)
    gt_merged_dir = Path(gt_merged_dir)

    # assert the directory to store the merged ground-truth masks is different than the original gt directory
    assert gt_merged_dir != "ground_truth"

    # Merge ground-truth masks for each corresponding images and store into the 'gt_merged_dir' folder
    if (root / gt_merged_dir).is_dir():
        logger.info(f"Found the directory of the merged ground-truth masks: {root / gt_merged_dir!s}")
    else:
        logger.info("Merging the multiple ground-truth masks for each corresponding images.")
        _merge_gt_mask(root, gt_merged_dir=gt_merged_dir)

    # Retrieve the image and mask files
    samples_list = []
    for f in root.glob("**/*"):
        if f.suffix in extensions:
            parts = f.parts
            # Ignore original 'ground_truth' folder because the 'gt_merged_dir' is used instead
            if "ground_truth" not in parts:
                split_folder, label_folder, image_path = parts[-3:]
                samples_list.append((str(root), split_folder, label_folder, image_path))

    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Replace validation to Split.VAL.value in the split column
    samples["split"] = samples["split"].replace("validation", Split.VAL.value)

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # separate masks from samples
    mask_samples = samples.loc[samples.split == str(gt_merged_dir)].sort_values(by="image_path", ignore_index=True)
    samples = samples[samples.split != str(gt_merged_dir)].sort_values(by="image_path", ignore_index=True)

    # assign mask paths to anomalous test images
    samples["mask_path"] = ""
    samples.loc[
        (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
        "mask_path",
    ] = mask_samples.image_path.to_numpy()

    # assert that the right mask files are associated with the right test images
    if len(samples.loc[samples.label_index == LabelName.ABNORMAL]):
        assert (
            samples.loc[samples.label_index == LabelName.ABNORMAL]
            .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
            .all()
        ), f"Mismatch between anomalous images and ground truth masks. Make sure the mask files in '{gt_merged_dir!s}' \
                folder follow the same naming convention as the anomalous images in the dataset (e.g. image: \
                '000.png', mask: '000.png')."

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


class MVTecLocoDataset(AnomalibDataset):
    """MVTec LOCO dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``.
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        root (Path | str): Path to the root of the dataset.
            Defaults to ``./datasets/MVTec_LOCO``.
        category (str): Sub-category of the dataset, e.g. 'breakfast_box'
            Defaults to ``breakfast_box``.
        split (str | Split | None): Split of the dataset, Split.TRAIN, Split.VAL, or Split.TEST
            Defaults to ``None``.

    Examples:
        .. code-block:: python

            from anomalib.data.image.mvtec_loco import MVTecLocoDataset
            from anomalib.data.utils.transforms import get_transforms

            transform = get_transforms(image_size=256)
            dataset = MVTecLocoDataset(
                task="classification",
                transform=transform,
                root='./datasets/MVTec_LOCO',
                category='breakfast_box',
            )
            dataset.setup()
            print(dataset[0].keys())
            # Output: dict_keys(['image_path', 'label', 'image'])

        When the task is segmentation, the dataset will also contain the mask:

        .. code-block:: python

            dataset.task = "segmentation"
            dataset.setup()
            print(dataset[0].keys())
            # Output: dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

        The image is a torch tensor of shape (C, H, W) and the mask is a torch tensor of shape (H, W).

        .. code-block:: python

            print(dataset[0]["image"].shape, dataset[0]["mask"].shape)
            # Output: (torch.Size([3, 256, 256]), torch.Size([256, 256]))
    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        root: Path | str = "./datasets/MVTec_LOCO",
        category: str = "breakfast_box",
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root_category = Path(root) / Path(category)
        self.split = split

    def _setup(self) -> None:
        self.samples = make_mvtec_loco_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
            gt_merged_dir=GT_MERGED_DIR,
        )


class MVTecLoco(AnomalibDataModule):
    """MVTec LOCO Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/MVTec_LOCO"``.
        category (str): Category of the MVTec LOCO dataset (e.g. "breakfast_box").
            Defaults to ``"breakfast_box"``.
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to ``(256, 256)``.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
            Defaults to ``None``.
        normalization (InputNormalizationMethod | str): Normalization method to be applied to the input images.
            Defaults to ``InputNormalizationMethod.IMAGENET``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
            Defaults to ``TaskType.SEGMENTATION``.
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing during training.
            Defaults to ``None``.
        transform_config_val (str | A.Compose | None, optional): Config for pre-processing
            during validation.
            Defaults to ``None``.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.FROM_DIR``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defaults to ``None``.

    Examples:
        To create an MVTec LOCO AD datamodule with default settings:

        >>> datamodule = MVTecLoco(root="anomalib/datasets/MVTec_LOCO")
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])

        To change the category of the dataset:

        >>> datamodule = MVTecLoco(category="pushpins")

        To change the image and batch size:

        >>> datamodule = MVTecLoco(image_size=(512, 512), train_batch_size=16, eval_batch_size=8)

        MVTec LOCO AD dataset provide an independent validation set with normal images only in the 'validation' folder.
        If you would like to use a different validation set splitted from train or test set,
        you can use the ``val_split_mode`` and ``val_split_ratio`` arguments to create a new validation set.

        >>> datamodule = MVTecLoco(val_split_mode=ValSplitMode.FROM_TEST, val_split_ratio=0.1)

        This will subsample the test set by 10% and use it as the validation set.
        If you would like to create a validation set synthetically that would
        not change the test set, you can use the ``ValSplitMode.SYNTHETIC`` option.

        >>> datamodule = MVTecLoco(val_split_mode=ValSplitMode.SYNTHETIC, val_split_ratio=0.2)
    """

    def __init__(
        self,
        root: Path | str = "./datasets/MVTec_LOCO",
        category: str = "breakfast_box",
        image_size: int | tuple[int, int] = (256, 256),
        center_crop: int | tuple[int, int] | None = None,
        normalization: InputNormalizationMethod | str = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.SEGMENTATION,
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.FROM_DIR,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = Path(category)

        transform_train = get_transforms(
            config=transform_config_train,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )
        transform_eval = get_transforms(
            config=transform_config_eval,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )

        self.train_data = MVTecLocoDataset(
            task=task,
            transform=transform_train,
            split=Split.TRAIN,
            root=root,
            category=category,
        )
        self.val_data = MVTecLocoDataset(
            task=task,
            transform=transform_eval,
            split=Split.VAL,
            root=root,
            category=category,
        )
        self.test_data = MVTecLocoDataset(
            task=task,
            transform=transform_eval,
            split=Split.TEST,
            root=root,
            category=category,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file system.
        If not, it downloads and extracts the dataset into the appropriate directory.

        Example:
            Assume the dataset is not available on the file system.
            Here's how the directory structure looks before and after calling the
            `prepare_data` method:

            Before:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                └── dataset2

            Calling the method:

            .. code-block:: python

                >> datamodule = MVTecLoco(root="./datasets/MVTec_LOCO", category="breakfast_box")
                >> datamodule.prepare_data()

            After:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                ├── dataset2
                └── MVTec_LOCO
                    ├── breakfast_box
                    ├── ...
                    └── splicing_connectors
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method overrides the parent class's method to also setup the val dataset.
        The MVTec LOCO dataset provides an independent validation subset.
        """
        assert self.train_data is not None
        assert self.val_data is not None
        assert self.test_data is not None

        self.train_data.setup()
        self.val_data.setup()
        self.test_data.setup()

        self._create_test_split()
        self._create_val_split()
