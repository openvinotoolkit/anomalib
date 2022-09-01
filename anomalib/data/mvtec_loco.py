"""

distinguishes structural and logical anomalies
n_image: 3644
splits:
no overlap and fixed
train
normal-only
n_image_train: 1772
validation
normal-only
n_image_validation: 304
test
normal + anomalous (structural and logical)
n_image_test: 1568
n_category: 5
breakfast_box
juice_bottle
pushpins
screw_bag
splicing_connectors
n_defect_type: 89

##################################################

configs
https://docs.google.com/spreadsheets/d/1qHbyTsU2At1fusQsV8KH3_qdp3SYHHLy7QtpohKJIk0/edit?usp=sharing

stats overview
https://docs.google.com/spreadsheets/d/11GSf1SVsHFYDSwMAULEd7g5QK7P3Y21YMB10D_g0-Gk/edit?usp=sharing

##################################################

assumptions
objects are in a fixed position (mechanical alignment)
illumination is well suited
the access to images with real anomalies is limited (“impossible”)
images only show a single object or logically ensemble set of objects (i.e. one-class setting although a “class” here is a composed object)
no training annotations -- although it is assumed that the images in training are indeed from the target class (i.e. no noise)
problem 1 (image-wise anomaly detection): “is there an anomaly in the image?”
problem 2 (pixel-wise anomaly detection or anomaly segmentation): “which pixels belong to the anomaly?”
pixel-wise metric: Saturated Per-Region Overlap  (sPRO) 
structural anomaly pixel annotation policy
defects are confined to local regions
each pixel that introduces a visual structure that is not present in the anomaly-free images is anomalous
logical anomaly pixel annotation policy
the union of all areas of the image that could be the cause for the anomaly is anomalous
a method is not necessarily required to predict the whole ground truth area as anomalous

##################################################

breakfast box
n_anomaly_type (n_structural, n_logical): 22 (5, 17)
logical constraints
contains 2 tangerines
contains 1 nectarine 
the tangerines and the nectarine on the left 
cereals (C) and a mix of banana chips and almonds (B&A) on the right
the ratio between C and B&A is fixed 
the relative position of C and B&A is fixed 
examples of logical defects
too many banana chips and almonds

##################################################

juice bottle
n_anomaly_type (n_structural, n_logical): 18 (7, 11)
logical constraints
there is 1 bottle 
the bottle is filled with a liquid and the fill level is always the same
the liquid is of 1 out of 3 colors (red, yellow, white-ish)
the bottle carries 2 labels
the first label is attached to the center of the bottle
the first label displays an icon that determines the type of liquid (cherry, orange, banana)
cherry: red
orange: yellow
banana: white-ish 
the second label is attached to the lower part of the bottle 
the second label contains the text “100% Juice”
examples of logical defects
(left) the icon does not match the type of juice
(middle) the icon is slightly misplaced
(right) the fill level is too high

##################################################

pushpins
n_anomaly_type (n_structural, n_logical): 8 (4, 4)
logical constraints
each compartment contains 1 pushpin 
examples of logical defects
1 compartment has a missing pin

##################################################

screw bag
n_anomaly_type (n_structural, n_logical): 20 (4, 16)
logical constraints
the bag contains 
2 washers
2 nuts
1 long screw
2 short screw
examples of logical defects
two long screws and lacks a short one

##################################################

splicing connectors
n_anomaly_type (n_structural, n_logical): 21 (8, 13)
logical constraints
there are 2 splicing connectors
they have the same number of cable clamps
they are linked by 1 cable
the number of clamps has a one-to-one correspondence to the color of the cable
2: yellow
3: blue
5: red
the cable has to terminate in the same relative position on its two ends such that the whole construction exhibits a mirror symmetry
examples of logical defects
(left) the two splicing connectors do not have the same number of clamps
(center) the color of the cable does not match the number of clamps
(right) the cable terminates in different positions 

##################################################

missing objects
the area in which the object could occur
the saturation threshold is chosen to be equal to the area of the missing object
the saturation threshold for an object is chosen from the lower end of the distribution of its (manually annotated) area
example (image): pushpin 
the missing pushpin can occur anywhere inside its compartment, therefore its entire area is annotated
the saturation threshold is set to the size of a pushpin 

##################################################

additional objects
too many instances of an object: all instances of the object are annotated
the saturation threshold is set to the area of the extraneous objects
example (image): splicing connectors
an additional cable is present between the two splicing connectors
it is not clear which of the two cables represents the anomaly, therefore both are annotated
the saturation threshold is set to the area of one cable (i.e., half of the annotated region)
properties
a method can obtain a perfect score even if it only marks one of the two cables as an anomaly 
a method that marks both is neither penalized nor (extra-)rewarded  

##################################################

other logical constraints 
example (image, left): juice bottle
the bottle is filled with orange juice but carries the label of the cherry juice
both the orange juice and the label with the cherry are present in the training set, but the logical anomaly arises due to the erroneous combination of the two in the same image
either the area filled with juice or the cherry as could be considered anomalous, therefore the union of the two regions is annotated
the saturation threshold is set to the area of the cherry because the segmentation of the cherry is sufficient to solve the anomaly localization

"""


"""
category	anomaly	type	gt_value	saturation_definition	saturation_parameter
breakfast_box	missing_almonds	logical	255	relative_to_anomaly	1.0000000
breakfast_box	missing_bananas	logical	254	relative_to_anomaly	1.0000000
breakfast_box	missing_toppings	logical	253	relative_to_anomaly	1.0000000
breakfast_box	missing_cereals	logical	252	relative_to_anomaly	1.0000000
breakfast_box	missing_cereals_and_toppings	logical	251	relative_to_anomaly	1.0000000
breakfast_box	nectarines_2_tangerine_1	logical	250	relative_to_image	0.0488770
breakfast_box	nectarine_1_tangerine_1	logical	249	relative_to_image	0.0411621
breakfast_box	nectarines_0_tangerines_2	logical	248	relative_to_image	0.0488770
breakfast_box	nectarines_0_tangerines_3	logical	247	relative_to_image	0.0488770
breakfast_box	nectarines_3_tangerines_0	logical	246	relative_to_image	0.0977539
breakfast_box	nectarines_0_tangerine_1	logical	245	relative_to_image	0.0900391
breakfast_box	nectarines_0_tangerines_0	logical	244	relative_to_image	0.1312012
breakfast_box	nectarines_0_tangerines_4	logical	243	relative_to_image	0.0823242
breakfast_box	compartments_swapped	logical	242	relative_to_anomaly	1.0000000
breakfast_box	overflow	logical	241	relative_to_anomaly	1.0000000
breakfast_box	underflow	logical	240	relative_to_anomaly	1.0000000
breakfast_box	wrong_ratio	logical	239	relative_to_anomaly	1.0000000
breakfast_box	mixed_cereals	structural	238	relative_to_anomaly	1.0000000
breakfast_box	fruit_damaged	structural	237	relative_to_anomaly	1.0000000
breakfast_box	box_damaged	structural	236	relative_to_anomaly	1.0000000
breakfast_box	toppings_crushed	structural	235	relative_to_anomaly	1.0000000
breakfast_box	contamination	structural	234	relative_to_anomaly	1.0000000
juice_box	missing_top_label	logical	255	relative_to_image	0.0550000
juice_box	missing_bottom_label	logical	254	relative_to_image	0.0255469
juice_box	swapped_labels	logical	253	relative_to_image	0.1100000
juice_box	damaged_label	structural	252	relative_to_anomaly	1.0000000
juice_box	rotated_label	structural	251	relative_to_anomaly	1.0000000
juice_box	misplaced_label_top	logical	250	relative_to_image	0.0550000
juice_box	misplaced_label_bottom	logical	249	relative_to_image	0.0255469
juice_box	label_text_incomplete	structural	248	relative_to_anomaly	1.0000000
juice_box	empty_bottle	logical	247	relative_to_anomaly	1.0000000
juice_box	wrong_fill_level_too_much	logical	246	relative_to_anomaly	1.0000000
juice_box	wrong_fill_level_not_enough	logical	245	relative_to_anomaly	1.0000000
juice_box	misplaced_fruit_icon	logical	244	relative_to_anomaly	1.0000000
juice_box	missing_fruit_icon	logical	243	relative_to_anomaly	1.0000000
juice_box	unknown_fruit_icon	structural	242	relative_to_anomaly	1.0000000
juice_box	incomplete_fruit_icon	structural	241	relative_to_anomaly	1.0000000
juice_box	wrong_juice_type	logical	240	relative_to_image	0.0035156
juice_box	juice_color	structural	239	relative_to_anomaly	1.0000000
juice_box	contamination	structural	238	relative_to_anomaly	1.0000000
pushpins	additional_1_pushpin	logical	255	relative_to_image	0.0037059
pushpins	additional_2_pushpins	logical	254	relative_to_image	0.0074118
pushpins	missing_pushpin	logical	253	relative_to_image	0.0037059
pushpins	missing_separator	logical	252	relative_to_anomaly	1.0000000
pushpins	front_bent	structural	251	relative_to_anomaly	1.0000000
pushpins	broken	structural	250	relative_to_anomaly	1.0000000
pushpins	color	structural	249	relative_to_anomaly	1.0000000
pushpins	contamination	structural	248	relative_to_anomaly	1.0000000
screw_bag	screw_too_long	logical	255	relative_to_image	0.0051136
screw_bag	screw_too_shor	logical	254	relative_to_image	0.0051136
screw_bag	screws_1_very_short	logical	253	relative_to_anomaly	1.0000000
screw_bag	screws_2_very_short	logical	252	relative_to_image	0.0102273
screw_bag	additional_1_long_screw	logical	251	relative_to_image	0.0168182
screw_bag	additional_1_short_screw	logical	250	relative_to_image	0.0117045
screw_bag	additional_1_nut_	logical	249	relative_to_image	0.0042614
screw_bag	additional_2_nuts_	logical	248	relative_to_image	0.0085227
screw_bag	additional_1_washer_	logical	247	relative_to_image	0.0031250
screw_bag	additional_2_washers_	logical	246	relative_to_image	0.0062500
screw_bag	missing_1_long_screw	logical	245	relative_to_image	0.0168182
screw_bag	missing_1_short_screw	logical	244	relative_to_image	0.0117045
screw_bag	missing_1_nut	logical	243	relative_to_image	0.0042614
screw_bag	missing_2_nuts	logical	242	relative_to_image	0.0085227
screw_bag	missing_1_washer	logical	241	relative_to_image	0.0031250
screw_bag	missing_2_washers	logical	240	relative_to_image	0.0062500
screw_bag	bag_broken	structural	239	relative_to_anomaly	1.0000000
screw_bag	color	structural	238	relative_to_anomaly	1.0000000
screw_bag	contamination	structural	237	relative_to_anomaly	1.0000000
screw_bag	part_broken	structural	236	relative_to_anomaly	1.0000000
splicing_connectors	wrong_connector_type_5_2	logical	255	relative_to_image	0.0464360
splicing_connectors	wrong_connector_type_5_3	logical	254	relative_to_image	0.0306574
splicing_connectors	wrong_connector_type_3_2	logical	253	relative_to_image	0.0152941
splicing_connectors	cable_too_short_t2	logical	252	relative_to_image	0.0368858
splicing_connectors	cable_too_short_t3	logical	251	relative_to_image	0.0526644
splicing_connectors	cable_too_short_t5	logical	250	relative_to_image	0.0830450
splicing_connectors	missing_connector	logical	249	relative_to_anomaly	1.0000000
splicing_connectors	missing_connector_and_cable	logical	248	relative_to_image	0.0716955
splicing_connectors	missing_cable	logical	247	relative_to_image	0.0124567
splicing_connectors	extra_cable	logical	246	relative_to_anomaly	0.5000000
splicing_connectors	cable_color	logical	245	relative_to_image	0.0124567
splicing_connectors	broken_cable	structural	244	relative_to_anomaly	1.0000000
splicing_connectors	cable_cut	logical	243	relative_to_anomaly	1.0000000
splicing_connectors	cable_not_plugged	structural	242	relative_to_anomaly	1.0000000
splicing_connectors	unknown_cable_color	structural	241	relative_to_anomaly	1.0000000
splicing_connectors	wrong_cable_location	logical	240	relative_to_image	0.0124567
splicing_connectors	flipped_connector	structural	239	relative_to_anomaly	1.0000000
splicing_connectors	broken_connector	structural	238	relative_to_anomaly	1.0000000
splicing_connectors	open_lever	structural	237	relative_to_anomaly	1.0000000
splicing_connectors	color	structural	236	relative_to_anomaly	1.0000000
splicing_connectors	contamination	structural	235	relative_to_anomaly	1.0000000
"""

import logging
import tarfile
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.request import urlretrieve

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import VisionDataset

from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import DownloadProgressBar, hash_check, read_image
from anomalib.data.utils.split import (
    create_validation_set_from_test_set,
    split_normal_images_in_train_set,
)
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(__name__)


def make_mvtec_loco_dataset(
    path: Path,
    split: Optional[str] = None,
    split_ratio: float = 0.1,
    seed: Optional[int] = None,
    create_validation_set: bool = False,
) -> DataFrame:
    samples_list = [(str(path),) + filename.parts[-3:] for filename in path.glob("**/*.png")]
    if len(samples_list) == 0:
        raise RuntimeError(f"Found 0 images in {path}")

    samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
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

    # Split the normal images in training set if test set doesn't
    # contain any normal images. This is needed because AUC score
    # cannot be computed based on 1-class
    if sum((samples.split == "test") & (samples.label == "good")) == 0:
        samples = split_normal_images_in_train_set(samples, split_ratio, seed)

    # Good images don't have mask
    samples.loc[(samples.split == "test") & (samples.label == "good"), "mask_path"] = ""

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = 0
    samples.loc[(samples.label != "good"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    if create_validation_set:
        samples = create_validation_set_from_test_set(samples, seed=seed)

    # Get the data frame for the split.
    if split is not None and split in ["train", "val", "test"]:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class MVTecLOCODataset(VisionDataset):
    """MVTec LOCO AD PyTorch Dataset."""

    def __init__(
        self,
        root: Union[Path, str],
        category: str,
        pre_process: PreProcessor,
        split: str,
        task: str = "segmentation",
        seed: Optional[int] = None,
        create_validation_set: bool = False,
    ) -> None:
        super().__init__(root)

        if seed is None:
            warnings.warn(
                "seed is None."
                " When seed is not set, images from the normal directory are split between training and test dir."
                " This will lead to inconsistency between runs."
            )

        self.root = Path(root) if isinstance(root, str) else root
        self.category: str = category
        self.split = split
        self.task = task

        self.pre_process = pre_process

        self.samples = make_mvtec_loco_dataset(
            path=self.root / category,
            split=self.split,
            seed=seed,
            create_validation_set=create_validation_set,
        )

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """
        item: Dict[str, Union[str, Tensor]] = {}

        image_path = self.samples.image_path[index]
        image = read_image(image_path)

        pre_processed = self.pre_process(image=image)
        item = {"image": pre_processed["image"]}

        if self.split in ["val", "test"]:
            label_index = self.samples.label_index[index]

            item["image_path"] = image_path
            item["label"] = label_index

            if self.task == "segmentation":
                mask_path = self.samples.mask_path[index]

                # Only Anomalous (1) images has masks in MVTec AD dataset.
                # Therefore, create empty mask for Normal (0) images.
                if label_index == 0:
                    mask = np.zeros(shape=image.shape[:2])
                else:
                    mask = cv2.imread(mask_path, flags=0) / 255.0

                pre_processed = self.pre_process(image=image, mask=mask)

                item["mask_path"] = mask_path
                item["image"] = pre_processed["image"]
                item["mask"] = pre_processed["mask"]

        return item


@DATAMODULE_REGISTRY
class MVTecLOCO(LightningDataModule):
    """MVTec LOCO AD Lightning Data Module."""

    def __init__(
        self,
        root: str,
        category: str,
        # TODO: Remove default values. IAAALD-211
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        task: str = "segmentation",
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_val: Optional[Union[str, A.Compose]] = None,
        seed: Optional[int] = None,
        create_validation_set: bool = False,
    ) -> None:
        super().__init__()

        self.root = root if isinstance(root, Path) else Path(root)
        self.category = category
        self.dataset_path = self.root / self.category
        self.transform_config_train = transform_config_train
        self.transform_config_val = transform_config_val
        self.image_size = image_size

        if self.transform_config_train is not None and self.transform_config_val is None:
            self.transform_config_val = self.transform_config_train

        self.pre_process_train = PreProcessor(config=self.transform_config_train, image_size=self.image_size)
        self.pre_process_val = PreProcessor(config=self.transform_config_val, image_size=self.image_size)

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.create_validation_set = create_validation_set
        self.task = task
        self.seed = seed

        self.train_data: Dataset
        self.test_data: Dataset
        if create_validation_set:
            self.val_data: Dataset
        self.inference_data: Dataset

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            self.root.mkdir(parents=True, exist_ok=True)

            logger.info("Downloading the Mvtec AD dataset.")
            url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094"
            dataset_name = "mvtec_anomaly_detection.tar.xz"
            zip_filename = self.root / dataset_name
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="MVTec AD") as progress_bar:
                urlretrieve(
                    url=f"{url}/{dataset_name}",
                    filename=zip_filename,
                    reporthook=progress_bar.update_to,
                )
            logger.info("Checking hash")
            hash_check(zip_filename, "eefca59f2cede9c3fc5b6befbfec275e")

            logger.info("Extracting the dataset.")
            with tarfile.open(zip_filename) as tar_file:
                tar_file.extractall(self.root)

            logger.info("Cleaning the tar file")
            (zip_filename).unlink()

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)

        """
        logger.info("Setting up train, validation, test and prediction datasets.")
        if stage in (None, "fit"):
            self.train_data = MVTecDataset(
                root=self.root,
                category=self.category,
                pre_process=self.pre_process_train,
                split="train",
                task=self.task,
                seed=self.seed,
                create_validation_set=self.create_validation_set,
            )

        if self.create_validation_set:
            self.val_data = MVTecDataset(
                root=self.root,
                category=self.category,
                pre_process=self.pre_process_val,
                split="val",
                task=self.task,
                seed=self.seed,
                create_validation_set=self.create_validation_set,
            )

        self.test_data = MVTecLOCODataset(
            root=self.root,
            category=self.category,
            pre_process=self.pre_process_val,
            split="test",
            task=self.task,
            seed=self.seed,
            create_validation_set=self.create_validation_set,
        )

        if stage == "predict":
            self.inference_data = InferenceDataset(
                path=self.root, image_size=self.image_size, transform_config=self.transform_config_val
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        dataset = self.val_data if self.create_validation_set else self.test_data
        return DataLoader(dataset=dataset, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(self.test_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Get predict dataloader."""
        return DataLoader(
            self.inference_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers
        )
