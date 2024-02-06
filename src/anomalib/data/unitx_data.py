"""UnitX Customer Per Defecttype Dataset.

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the UnitX per-defect type dataset.
"""
# %%
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import albumentations as A
from pandas import DataFrame

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import (
    InputNormalizationMethod,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    get_transforms,
)

logger = logging.getLogger(__name__)


KAIZHONG_CATEGORIES = (
    "NGK钩上面脏",
    "NGK钩上黑皮",
    "NGK钩侧黑皮",
    "NGK钩刷损",
    "NGK钩叠料",
    "NGK钩损",
    "NGK钩毛刺",
    "NGK钩表面刷损",
)


# %%
def make_unitx_dataset(
    root: str | Path,
    split: str | Split | None = None,
    exclude_labels: List[str] = [],
    exclude_asset_ids: List[str] = [],
) -> DataFrame:
    """Create MVTec AD samples by parsing the MVTec AD data file structure.

        The files are expected to follow the directory structure:

    <root>
    ├── good
    │  ├── 1643262694171-GOODxB84FOD     # Good images
    │  └── ...
    ├── <defect_name_1>, # i.e. NGK钩上黑皮
    │  ├── pred
    │  │  ├── 0.png
    │  ├── train
    │  │  ├── images
    │  │  │  ├── 1643262694171-HUDOB84FOD # file: PNG image data, 960 x 416, 8-bit/color RGB, non-interlaced
    │  │  │  └── ...
    │  │  └── masks
    │  │     ├── 1643262694171-HUDOB84FOD.npy # ground truth as np.array
    │  │     └── ...
    │  └── val
    │     ├── images
    │     │  └── ...
    │     └── masks
    │        └── ...
    ├── <defect_name_1>, # i.e. NGK钩侧黑皮

        These are the assumptions:
        - All the images and masks have the same height and width.
        - Images and masks have the same base filenames.
        - Good image has no empty mask

        This function creates a dataframe to store the parsed information based on the following format:
        |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
        |   | path          | split | label   | image_path    | mask_path                             | label_index |
        |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
        | 0 | datasets/name |  test | defect  |  filename     | mask.npy                              | 1           |
        |---|---------------|-------|---------|---------------|---------------------------------------|-------------|

        Args:
            path (Path): Path to dataset
            split (str | Split | None, optional): Dataset split (ie., either train or test). Defaults to None.
            exclude_labels: Labels to be excluded from the dataset.  i.e. "NGall"
            exclude_labels: Asset iDs to be excluded from the dataset. i.e. "1684998025670-8LS7AG8HN4"

        Examples:
            The following example shows how to get training samples as DataFrame:

            >>> split = "train"
            >>> root = f"/Users/steveyang/datasets/1416_by_ng"
            >>> samples = make_unitx_dataset(root, split=split, exclude_labels=['NGall'])
            >>> samples.head()

                path           label   split     image_path                 mask_path                     label_index
            0  /1416_by_ng  NGK钩上面脏  train  /1416_by_ng/NGK钩上面脏/t...  /1416_by_ng/NGK钩上面脏/t...            1
            1  /1416_by_ng   NGK钩刷损  train  /1416_by_ng/NGK钩刷损/tr...  /1416_by_ng/NGK钩刷损/tr...            1
            2  /1416_by_ng   NGK钩刷损  train  /1416_by_ng/NGK钩刷损/tr...  /1416_by_ng/NGall/tra...            1
            3  /1416_by_ng   NGK钩刷损  train  /1416_by_ng/NGK钩刷损/tr...  /1416_by_ng/NGK钩刷损/tr...            1
            4  /1416_by_ng   NGK钩刷损  train  /1416_by_ng/NGK钩刷损/tr...  /1416_by_ng/NGK钩刷损/tr...            1

        Returns:
            DataFrame: an output dataframe containing the samples of the dataset.
    """
    root = Path(root)
    samples_list = [
        (str(root),) + f.parts[-4:]
        for f in root.glob(f"*/*/images/*")
        if f.is_file() and (f.parts[-3] == "train" or f.parts[-3] == "val")
    ]
    if not samples_list:
        raise RuntimeError(f"Found 0 images in {root}")

    samples = DataFrame(samples_list, columns=["path", "label", "split", "is_images_or_masks", "asset_id"])
    # separate masks from samples
    # mask_samples = samples[samples.is_images_or_masks == "masks"].sort_values(by="asset_id", ignore_index=True)
    # mask_samples['asset_id'] = mask_samples['asset_id'].apply(lambda x: x.strip(".npy"))
    # samples = samples[samples.is_images_or_masks == "images"].sort_values(by="asset_id", ignore_index=True)
    # assert(len(mask_samples) == len(samples))

    samples = samples[~samples.label.isin(exclude_labels)]
    samples = samples[~samples.asset_id.isin(exclude_asset_ids)]

    # mask_path, image_path using absolute path
    samples["image_path"] = (
        samples.path + "/" + samples.label + "/" + samples.split + "/" + "images" + "/" + samples.asset_id
    )

    samples["mask_path"] = (
        samples.path
        + "/"
        + samples.label
        + "/"
        + samples.split
        + "/"
        + "masks"
        + "/"
        + samples.asset_id
    )
    samples["mask_path"] = samples["mask_path"].apply(lambda x: x.replace(".png", "") + ".npy") # add back the suffix

    samples = samples.drop(columns=["is_images_or_masks", "asset_id"])

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "OKOK"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "OKOK"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)


    # Validation of the DataFrame
    assert all(samples.image_path.notnull()), "Some image paths are null."
    assert all(samples.mask_path.notnull()), "Some mask paths are null."

    # assert that the right mask files are associated with the right test images
    if len(samples.loc[samples.label_index == LabelName.ABNORMAL]):
        for _, row in samples.loc[samples.label_index == LabelName.ABNORMAL].iterrows():
            if not Path(row["mask_path"]).exists():
                print(f"Mask path {row['mask_path']} does not exist.")
            # assert Path(row['mask_path']).exists(), f"Mask path {row['mask_path']} does not exist."
        assert (
            samples.loc[samples.label_index == LabelName.ABNORMAL]
            .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
            .all()
        ), "Mismatch between anomalous images and ground truth masks. Make sure the mask files in 'ground_truth' \
                folder follow the same naming convention as the anomalous images in the dataset (e.g. image: \
                '000.png', mask: '000.png' or '000_mask.png')."

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


class UnitXPerDefectDataset(AnomalibDataset):
    """UnitX per-defect dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
        root (Path | str): Path to the root of the dataset
    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        root: Path | str,
        split: str | Split | None = None,
        exclude_labels: List[str] | None = None,
        exclude_asset_ids: List[str] | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root_category = Path(root)
        self.split = split
        self.exclude_labels = exclude_labels
        self.exclude_asset_ids = exclude_asset_ids

    def _setup(self) -> None:
        self.samples = make_unitx_dataset(
            self.root_category,
            split=self.split,
            exclude_labels=self.exclude_labels,
            exclude_asset_ids=self.exclude_asset_ids,
        )


class UnitXPerDefect(AnomalibDataModule):
    """MVTec Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to None.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
        normalize (bool): When True, the images will be normalized to the ImageNet statistics.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (str | A.Compose | None, optional): Config for pre-processing
            during validation.
            Defaults to None.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
    """

    def __init__(
        self,
        root: Path | str,
        exclude_labels: List[str] | None = None,
        exclude_asset_ids: List[str] | None = None,
        image_size: int | tuple[int, int] | None = None,
        center_crop: int | tuple[int, int] | None = None,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.SEGMENTATION,
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
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

        self.train_data = UnitXPerDefectDataset(
            task=task, 
            transform=transform_train, 
            split=Split.VAL, 
            root=root,
            exclude_labels=exclude_labels,
            exclude_asset_ids=exclude_asset_ids,
        )

        self.test_data = UnitXPerDefectDataset(
            task=task, 
            transform=transform_eval, 
            split=Split.TRAIN, 
            root=root,
            exclude_labels=[],
            exclude_asset_ids=exclude_asset_ids,
        )

        self.val_data = self.test_data
