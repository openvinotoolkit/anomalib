from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as A
from pandas import DataFrame

from anomalib.data.base import AnomalibDataModule, AnomalibDataset, Split, ValSplitMode
from anomalib.data.utils.split import split_normals_and_anomalous
from anomalib.pre_processing import PreProcessor


def make_mvtec_dataset(root: Union[str, Path], split: Split = Split.FULL) -> DataFrame:
    """Create MVTec AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
    |   | path          | split | label   | image_path    | mask_path                             | label_index |
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
    | 0 | datasets/name |  test |  defect |  filename.png | ground_truth/defect/filename_mask.png | 1           |
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|

    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.
    """
    samples_list = [(str(root),) + filename.parts[-3:] for filename in Path(root).glob("**/*.png")]
    if len(samples_list) == 0:
        raise RuntimeError(f"Found 0 images in {root}")

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
    samples = samples[samples.split != "ground_truth"]

    # Create mask_path column
    samples["mask_path"] = (
        samples.path
        + "/ground_truth/"
        + samples.label
        + "/"
        + samples.image_path.str.rstrip("png").str.rstrip(".")
        + "_mask.png"
    )

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Good images don't have mask
    samples.loc[(samples.split == "test") & (samples.label == "good"), "mask_path"] = ""

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = 0
    samples.loc[(samples.label != "good"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    if split != Split.FULL:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


class MVTec(AnomalibDataset):
    def __init__(self, task: str, pre_process: PreProcessor, split: Split, root, category, samples=None) -> None:
        super().__init__(task=task, pre_process=pre_process, samples=samples)

        self.root_category = Path(root) / Path(category)
        self.split = split

    def _setup(self):
        self._samples = make_mvtec_dataset(self.root_category, split=self.split)


class MVTecDataModule(AnomalibDataModule):
    def __init__(
        self,
        root: str,
        category: str,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        task: str = "segmentation",
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_val: Optional[Union[str, A.Compose]] = None,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
    ):
        super().__init__(
            task=task,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            val_split_mode=val_split_mode,
        )

        self.val_split_mode = val_split_mode

        pre_process_train = PreProcessor(config=transform_config_train, image_size=image_size)
        pre_process_infer = PreProcessor(config=transform_config_val, image_size=image_size)

        self.train_data = MVTec(
            task=task, pre_process=pre_process_train, split=Split.TRAIN, root=root, category=category
        )
        self.test_data = MVTec(task=task, pre_process=pre_process_infer, split=Split.TEST, root=root, category=category)

    def _setup(self, _stage: Optional[str] = None) -> None:
        """Set up the datasets and perform dynamic subset splitting if necessary.

        This method may be overridden in subclasses for custom splitting behaviour.
        """
        assert self.train_data is not None
        assert self.test_data is not None

        self.train_data.setup()
        self.test_data.setup()
        if self.val_split_mode == ValSplitMode.FROM_TEST:
            self.val_data, self.test_data = split_normals_and_anomalous(self.test_data, 0.5)
        elif self.val_split_mode == ValSplitMode.SAME_AS_TEST:
            self.val_data = self.test_data
        else:
            raise ValueError(f"Unknown validation split mode: {self.val_split_mode}")
