"""
Dataset Helpers for OTE Training
"""

from pathlib import Path
from typing import List, Union

from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.image import Image
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from pandas.core.frame import DataFrame
from tqdm import tqdm

from anomalib.datasets.anomaly_dataset import make_dataset


class OTEAnomalyDatasetGenerator:
    """
    Generate OTE Dataset from the anomaly detection datasets that follows the MVTec format.
    Args:
        path (Union[str, Path], optional): Path to the MVTec dataset category.
            Defaults to "./datasets/MVTec/bottle".
        split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.5.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
        create_validation_set (bool, optional): Create validation set from the test set by splitting
            it to half. Default to True.

    Examples:
        >>> dataset_generator = OTEAnomalyDatasetGenerator()
        >>> dataset = dataset_generator.generate()
        >>> dataset[0].media.numpy.shape
        (900, 900, 3)
    """

    def __init__(
        self,
        path: Union[str, Path] = "./datasets/MVTec/bottle",
        split_ratio: float = 0.5,
        seed: int = 0,
        create_validation_set: bool = True,
    ):
        self.path = path if isinstance(path, Path) else Path(path)
        self.split_ratio = split_ratio
        self.seed = seed
        self.create_validation_set = create_validation_set

        self.normal_label = LabelEntity(name="normal", domain=Domain.ANOMALY_CLASSIFICATION)
        self.abnormal_label = LabelEntity(name="anomalous", domain=Domain.ANOMALY_CLASSIFICATION)

    def get_samples(self) -> DataFrame:
        """
        Get MVTec samples in a pandas DataFrame. Update the certain columns
        to match the OTE naming terminology. For example, column `split` is
        renamed to `subset`. Labels are also renamed by creating their
        corresponding OTE LabelEntities

        Returns:
            DataFrame: Final list of samples comprising all the required
                information to create the OTE Dataset.
        """
        samples = make_dataset(self.path, self.split_ratio, self.seed, self.create_validation_set)

        # Set the OTE SDK Splits
        samples = samples.rename(columns={"split": "subset"})
        samples.loc[samples.subset == "train", "subset"] = Subset.TRAINING
        samples.loc[samples.subset == "val", "subset"] = Subset.VALIDATION
        samples.loc[samples.subset == "test", "subset"] = Subset.TESTING

        # Create and Set the OTE Labels
        samples.loc[samples.label != "good", "label"] = self.abnormal_label
        samples.loc[samples.label == "good", "label"] = self.normal_label

        samples = samples.reset_index(drop=True)

        return samples

    def generate(self) -> DatasetEntity:
        """
        Generate OTE Anomaly Dataset

        Returns:
            DatasetEntity: Output OTE Anomaly Dataset from an MVTec
        """
        samples = self.get_samples()
        dataset_items: List[DatasetItemEntity] = []
        for _, sample in tqdm(samples.iterrows()):
            # Create image
            image = Image(file_path=sample.image_path)

            # Create annotation
            shape = Rectangle(x1=0, y1=0, x2=1, y2=1)
            labels = [ScoredLabel(sample.label)]
            annotations = [Annotation(shape=shape, labels=labels)]
            annotation_scene = AnnotationSceneEntity(annotations=annotations, kind=AnnotationSceneKind.ANNOTATION)

            # Create dataset item
            dataset_item = DatasetItemEntity(media=image, annotation_scene=annotation_scene, subset=sample.subset)

            # Add to dataset items
            dataset_items.append(dataset_item)

        dataset = DatasetEntity(items=dataset_items)
        return dataset
