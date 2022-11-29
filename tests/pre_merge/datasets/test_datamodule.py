"""Test Dataset."""

import os

import numpy as np
import pytest

from anomalib.config import update_input_size_config
from anomalib.data import Avenue, BTech, Folder, MVTec, UCSDped, get_datamodule
from anomalib.pre_processing.transforms import Denormalize, ToNumpy
from tests.helpers.config import get_test_configurable_parameters
from tests.helpers.dataset import TestDataset, get_dataset_path

EXPECTED_KEYS_CLASSIFICATION = ["image", "label"]
EXPECTED_KEYS_DETECTION = ["image", "label", "boxes"]
EXPECTED_KEYS_SEGMENTATION = ["image", "label", "mask"]
EXPECTED_KEYS_PER_TASK_TYPE = {
    "classification": EXPECTED_KEYS_CLASSIFICATION,
    "detection": EXPECTED_KEYS_DETECTION,
    "segmentation": EXPECTED_KEYS_SEGMENTATION,
}


@pytest.fixture(autouse=True)
def make_avenue_data_module():
    def make(task="classification", batch_size=1):
        root = get_dataset_path(dataset="avenue")
        data_module = Avenue(
            root=root,
            gt_dir=os.path.join(root, "ground_truth_demo"),
            image_size=(256, 256),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=0,
            task=task,
            val_split_mode="from_test",
        )
        data_module.setup()
        return data_module

    return make


@pytest.fixture(autouse=True)
def make_mvtec_data_module():
    def make(task="classification", batch_size=1):
        data_module = MVTec(
            root=get_dataset_path(dataset="MVTec"),
            category="leather",
            image_size=(256, 256),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=0,
            task=task,
            val_split_mode="from_test",
        )
        data_module.prepare_data()
        data_module.setup()
        return data_module

    return make


@pytest.fixture(autouse=True)
def make_btech_data_module():
    def make(task="classification", batch_size=1):
        """Create BTech Data Module."""
        data_module = BTech(
            root=get_dataset_path(dataset="BTech"),
            category="01",
            image_size=(256, 256),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=0,
            task=task,
            val_split_mode="from_test",
        )
        data_module.prepare_data()
        data_module.setup()
        return data_module

    return make


@pytest.fixture(autouse=True)
def make_folder_data_module():
    def make(task="classification", batch_size=1):
        """Create Folder Data Module."""
        root = get_dataset_path(dataset="bottle")
        data_module = Folder(
            root=root,
            normal_dir="good",
            abnormal_dir="broken_large",
            mask_dir=os.path.join(root, "ground_truth/broken_large"),
            split_ratio=0.2,
            image_size=(256, 256),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=8,
            task=task,
            val_split_mode="from_test",
        )
        data_module.setup()
        return data_module

    return make


@pytest.fixture(autouse=True)
def make_ucsdped_data_module():
    def make(task="classification", batch_size=1):
        data_module = UCSDped(
            root=get_dataset_path(dataset="ucsd"),
            category="UCSDped2",
            image_size=(256, 256),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=0,
            task=task,
            val_split_mode="from_test",
        )
        data_module.setup()
        return data_module

    return make


@pytest.fixture(autouse=True)
def make_data_sample():
    root = get_dataset_path(dataset="bottle")
    datamodule = Folder(
        root=root,
        normal_dir="good",
        abnormal_dir="broken_large",
        mask_dir=os.path.join(root, "ground_truth/broken_large"),
        split_ratio=0.2,
        image_size=(256, 256),
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=8,
        task="classification",
        val_split_mode="from_test",
    )
    datamodule.setup()
    _, data = next(enumerate(datamodule.train_dataloader()))
    return data


@pytest.mark.parametrize(
    "make_data_module",
    [
        "make_mvtec_data_module",
        "make_btech_data_module",
        "make_folder_data_module",
        "make_avenue_data_module",
        "make_ucsdped_data_module",
    ],
)
class TestDataModule:
    """Test MVTec AD Data Module."""

    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_batch_size(self, make_data_module, batch_size, request):
        """test_mvtec_datamodule [summary]"""
        data_module = request.getfixturevalue(make_data_module)(batch_size=batch_size)
        _, train_data_sample = next(enumerate(data_module.train_dataloader()))
        _, val_data_sample = next(enumerate(data_module.val_dataloader()))
        assert train_data_sample["image"].shape[0] == batch_size
        assert val_data_sample["image"].shape[0] == batch_size

    @pytest.mark.parametrize("task", ["classification", "detection", "segmentation"])
    def test_val_and_test_dataloaders_has_mask_and_gt(self, make_data_module, task, request):
        """Test Validation and Test dataloaders should return filenames, image, mask and label."""
        data_module = request.getfixturevalue(make_data_module)(task=task)
        _, val_data = next(enumerate(data_module.val_dataloader()))
        _, test_data = next(enumerate(data_module.test_dataloader()))

        assert set(EXPECTED_KEYS_PER_TASK_TYPE[task]).issubset(set(val_data.keys()))
        assert set(EXPECTED_KEYS_PER_TASK_TYPE[task]).issubset(set(test_data.keys()))

    def test_non_overlapping_splits(self, make_data_module, request):
        """This test ensures that the train and test splits generated are non-overlapping."""
        data_module = request.getfixturevalue(make_data_module)()
        assert (
            len(
                set(data_module.test_data.samples["image_path"].values).intersection(
                    set(data_module.train_data.samples["image_path"].values)
                )
            )
            == 0
        ), "Found train and test split contamination"


class TestDenormalize:
    """Test Denormalize Util."""

    def test_denormalize_image_pixel_values(self, data_sample):
        """Test Denormalize denormalizes tensor into [0, 256] range."""
        denormalized_sample = Denormalize()(data_sample["image"].squeeze())
        assert denormalized_sample.min() >= 0 and denormalized_sample.max() <= 256

    def test_denormalize_return_numpy(self, data_sample):
        """Denormalize should return a numpy array."""
        denormalized_sample = Denormalize()(data_sample["image"].squeeze())
        assert isinstance(denormalized_sample, np.ndarray)

    def test_denormalize_channel_order(self, data_sample):
        """Denormalize should return a numpy array of order [HxWxC]"""
        denormalized_sample = Denormalize()(data_sample["image"].squeeze())
        assert len(denormalized_sample.shape) == 3 and denormalized_sample.shape[-1] == 3

    def test_representation(self):
        """Test Denormalize representation should return string
        Denormalize()"""
        assert str(Denormalize()) == "Denormalize()"


class TestToNumpy:
    """Test ToNumpy whether it properly converts tensor into numpy array."""

    def test_to_numpy_image_pixel_values(self, data_sample):
        """Test ToNumpy should return an array whose pixels in the range of [0,
        256]"""
        array = ToNumpy()(data_sample["image"])
        assert array.min() >= 0 and array.max() <= 256

    def test_to_numpy_converts_tensor_to_np_array(self, data_sample):
        """ToNumpy returns a numpy array."""
        array = ToNumpy()(data_sample["image"])
        assert isinstance(array, np.ndarray)

    def test_to_numpy_channel_order(self, data_sample):
        """ToNumpy() should return a numpy array of order [HxWxC]"""
        array = ToNumpy()(data_sample["image"])
        assert len(array.shape) == 3 and array.shape[-1] == 3

    def test_one_channel_images(self, data_sample):
        """One channel tensor should be converted to HxW np array."""
        data = data_sample["image"][:, 0, :, :].unsqueeze(0)
        array = ToNumpy()(data)
        assert len(array.shape) == 2

    def test_representation(self):
        """Test ToNumpy() representation should return string `ToNumpy()`"""
        assert str(ToNumpy()) == "ToNumpy()"


class TestConfigToDataModule:
    """Tests that check if the dataset parameters in the config achieve the desired effect."""

    @pytest.mark.parametrize(
        ["input_size", "effective_image_size"],
        [
            (512, (512, 512)),
            ((245, 276), (245, 276)),
            ((263, 134), (263, 134)),
            ((267, 267), (267, 267)),
        ],
    )
    @TestDataset(num_train=20, num_test=10)
    def test_image_size(self, input_size, effective_image_size, category="shapes", path=None):
        """Test if the image size parameter works as expected."""
        configurable_parameters = get_test_configurable_parameters(dataset_path=path, model_name="stfpm")
        configurable_parameters.dataset.category = category
        configurable_parameters.dataset.image_size = input_size
        configurable_parameters = update_input_size_config(configurable_parameters)

        data_module = get_datamodule(configurable_parameters)
        data_module.setup()
        assert next(iter(data_module.train_dataloader()))["image"].shape[-2:] == effective_image_size
