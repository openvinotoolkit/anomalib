"""CUHK Avenue Dataset."""

import logging
import zipfile
from collections import namedtuple
from os import listdir, rmdir
from os.path import join
from pathlib import Path
from shutil import move
from typing import Callable, Optional, Tuple, Union
from urllib.request import urlretrieve

import albumentations as A
import numpy as np
import scipy.io
from pandas import DataFrame
from torch import Tensor

from anomalib.data.base import AnomalibDataModule
from anomalib.data.base.video import VideoAnomalibDataset
from anomalib.data.utils import DownloadProgressBar, Split, ValSplitMode, hash_check
from anomalib.data.utils.video import ClipsIndexer
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(__name__)


# define some info used for downloading the dataset
DownloadInfo = namedtuple("DownloadInfo", "description zip_filename expected_hash extracted_folder_name")
DATASET_DOWNLOAD_INFO = DownloadInfo(
    "dataset", "Avenue_Dataset.zip", "b7a34b212ecdd30efbd989a6dcb1aceb", "Avenue Dataset"
)
ANNOTATIONS_DOWNLOAD_INFO = DownloadInfo(
    "annotations", "ground_truth_demo.zip", "e8e3bff99195b6b511534083b9dbe1f5", "ground_truth_demo"
)


def make_avenue_dataset(root: Path, gt_dir: Path, split: Optional[Union[Split, str]] = None):
    """Create CUHK Avenue dataset by parsing the file structure.

    The files are expected to follow the structure:
        path/to/dataset/[training_videos|testing_videos]/video_filename.avi
        path/to/ground_truth/mask_filename.mat

    Args:
        root (Path): Path to dataset
        gt_dir (Path): Path to the ground truth
        split (Optional[Union[Split, str]], optional): Dataset split (ie., either train or test). Defaults to None.

    Example:
        The following example shows how to get testing samples from Avenue dataset:

        >>> root = Path('./avenue')
        >>> gt_dir = Path('./avenue/masks')
        >>> samples = make_avenue_dataset(path, gt_dir, split='test')
        >>> samples.head()
           root     folder         image_path                      mask_path                   split
        0  ./avenue testing_videos ./avenue/training_videos/01.avi ./avenue/masks/01_label.mat test
        1  ./avenue testing_videos ./avenue/training_videos/02.avi ./avenue/masks/01_label.mat test
        ...

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """

    samples_list = [(str(root),) + filename.parts[-2:] for filename in Path(root).glob("**/*.avi")]
    samples = DataFrame(samples_list, columns=["root", "folder", "image_path"])

    samples.loc[samples.folder == "testing_videos", "mask_path"] = (
        samples.image_path.str.split(".").str[0].str.lstrip("0") + "_label.mat"
    )
    samples.loc[samples.folder == "testing_videos", "mask_path"] = str(gt_dir) + "/" + samples.mask_path
    samples.loc[samples.folder == "training_videos", "mask_path"] = ""

    samples["image_path"] = samples.root + "/" + samples.folder + "/" + samples.image_path

    samples.loc[samples.folder == "training_videos", "split"] = "train"
    samples.loc[samples.folder == "testing_videos", "split"] = "test"

    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class AvenueClipsIndexer(ClipsIndexer):
    """Clips class for UCSDped dataset."""

    def get_mask(self, idx) -> Optional[Tensor]:
        """Retrieve the masks from the file system."""

        video_idx, frames_idx = self.get_clip_location(idx)
        matfile = self.mask_paths[video_idx]
        if matfile == "":  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]

        mat = scipy.io.loadmat(matfile)
        masks = Tensor(np.vstack([np.stack(m) for m in mat["volLabel"]]))
        return masks[frames]


class AvenueDataset(VideoAnomalibDataset):
    """Avenue Dataset class.

    Args:
        task (str): Task type, either 'classification' or 'segmentation'
        root (str): Path to the root of the dataset
        gt_dir (str): Path to the ground truth files
        pre_process (PreProcessor): Pre-processor object
        split (Optional[Union[Split, str]]): Split of the dataset, usually Split.TRAIN or Split.TEST
        frames_per_clip (int, optional): Number of video frames in each clip.
        stride (int, optional): Number of frames between each consecutive video clip.
    """

    def __init__(
        self,
        task: str,
        root: Union[Path, str],
        gt_dir: str,
        pre_process: PreProcessor,
        split: Split,
        frames_per_clip: int = 1,
        stride: int = 1,
    ):
        super().__init__(task, pre_process, frames_per_clip, stride)

        self.root = root
        self.gt_dir = gt_dir
        self.split = split
        self.indexer_cls: Callable = AvenueClipsIndexer

    def _setup(self):
        """Create and assign samples."""
        self.samples = make_avenue_dataset(self.root, self.gt_dir, self.split)


class Avenue(AnomalibDataModule):
    """Avenue DataModule class.

    Args:
        root (str): Path to the root of the dataset
        gt_dir (str): Path to the ground truth files
        frames_per_clip (int, optional): Number of video frames in each clip.
        stride (int, optional): Number of frames between each consecutive video clip.
        task (str): Task type, either 'classification' or 'segmentation'
        image_size (Optional[Union[int, Tuple[int, int]]], optional): Size of the input image.
            Defaults to None.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        transform_config_train (Optional[Union[str, A.Compose]], optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (Optional[Union[str, A.Compose]], optional): Config for pre-processing
            during validation.
            Defaults to None.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
    """

    def __init__(
        self,
        root: str,
        gt_dir: str,
        frames_per_clip: int = 1,
        stride: int = 1,
        task: str = "segmentation",
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_eval: Optional[Union[str, A.Compose]] = None,
        val_split_mode: ValSplitMode = ValSplitMode.FROM_TEST,
    ):
        super().__init__(train_batch_size, eval_batch_size, num_workers, val_split_mode)

        self.root = Path(root)
        self.gt_dir = Path(gt_dir)

        pre_process_train = PreProcessor(config=transform_config_train, image_size=image_size)
        pre_process_eval = PreProcessor(config=transform_config_eval, image_size=image_size)

        self.train_data = AvenueDataset(
            task=task,
            pre_process=pre_process_train,
            frames_per_clip=frames_per_clip,
            stride=stride,
            root=root,
            gt_dir=gt_dir,
            split=Split.TRAIN,
        )

        self.test_data = AvenueDataset(
            task=task,
            pre_process=pre_process_eval,
            frames_per_clip=frames_per_clip,
            stride=stride,
            root=root,
            gt_dir=gt_dir,
            split=Split.TEST,
        )

    def prepare_data(self) -> None:
        """Download dataset."""
        self._download_and_extract(self.root, DATASET_DOWNLOAD_INFO)
        self._download_and_extract(self.gt_dir, ANNOTATIONS_DOWNLOAD_INFO)

    @staticmethod
    def _download_and_extract(root: Path, info: DownloadInfo):
        """Download and extract the dataset or the annotations.

        Args:
            root (Path): File system location where to search for or install the dataset.
            info (DownloadInfo): Contains download information specific to either dataset or ground truth annotations.
        """
        if root.is_dir():
            logger.info("Found the %s folder.", info.description)
        else:
            root.mkdir(parents=True, exist_ok=True)

            logger.info("Downloading %s.", info.description)
            url = "http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal"
            zip_filepath = root / info.zip_filename
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=info.description) as progress_bar:
                urlretrieve(
                    url=f"{url}/{info.zip_filename}",
                    filename=zip_filepath,
                    reporthook=progress_bar.update_to,
                )
            logger.info("Checking hash")
            hash_check(zip_filepath, info.expected_hash)

            logger.info("Extracting the %s.", info.description)
            with zipfile.ZipFile(zip_filepath, "r") as zip_file:
                zip_file.extractall(root)
            # move contents to root
            for filename in listdir(join(root, info.extracted_folder_name)):
                move(join(root, info.extracted_folder_name, filename), join(root, filename))
            rmdir(join(root, info.extracted_folder_name))

            logger.info("Cleaning the zip file")
            (zip_filepath).unlink()
