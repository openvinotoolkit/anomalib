"""Test Dataset."""

import os

import numpy as np
import pytest

from anomalib.config import update_input_size_config
from anomalib.data import (
    Avenue,
    BTech,
    Folder,
    MVTec,
    ShanghaiTech,
    UCSDped,
    Visa,
    get_datamodule,
)
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


def make_avenue_data_module(task="classification", batch_size=1, val_split_mode="from_test"):
    root = get_dataset_path(dataset="avenue")
    data_module = Avenue(
        root=root,
        gt_dir=os.path.join(root, "ground_truth_demo"),
        image_size=(256, 256),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        task=task,
        val_split_mode=val_split_mode,
    )
    data_module.setup()
    return data_module


def make_mvtec_data_module(task="classification", batch_size=1, test_split_mode="from_dir", val_split_mode="from_test"):
    data_module = MVTec(
        root=get_dataset_path(dataset="MVTec"),
        category="leather",
        image_size=(256, 256),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        task=task,
        test_split_mode=test_split_mode,
        val_split_mode=val_split_mode,
    )
    data_module.prepare_data()
    data_module.setup()
    return data_module


def make_btech_data_module(task="classification", batch_size=1, test_split_mode="from_dir", val_split_mode="from_test"):
    """Create BTech Data Module."""
    data_module = BTech(
        root=get_dataset_path(dataset="BTech"),
        category="01",
        image_size=(256, 256),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        task=task,
        test_split_mode=test_split_mode,
        val_split_mode=val_split_mode,
    )
    data_module.prepare_data()
    data_module.setup()
    return data_module


def make_folder_data_module(
    task="classification",
    batch_size=1,
    test_split_mode="from_dir",
    val_split_mode="from_test",
    normal_dir="good",
    abnormal_dir="broken_large",
    normal_test_dir="good_test",
    mask_dir="ground_truth/broken_large",
):
    """Create Folder Data Module."""
    root = get_dataset_path(dataset="bottle")
    data_module = Folder(
        root=root,
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir,
        normal_test_dir=normal_test_dir,
        mask_dir=mask_dir,
        normal_split_ratio=0.2,
        image_size=(256, 256),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        task=task,
        test_split_mode=test_split_mode,
        val_split_mode=val_split_mode,
    )
    data_module.setup()
    return data_module


def make_shanghaitech_data_module(task="classification", batch_size=1, val_split_mode="from_test"):
    data_module = ShanghaiTech(
        root=get_dataset_path(dataset="shanghaitech"),
        scene=1,
        image_size=(256, 256),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        task=task,
        val_split_mode=val_split_mode,
    )
    data_module.setup()
    return data_module


def make_ucsdped_data_module(task="classification", batch_size=1, val_split_mode="from_test"):
    data_module = UCSDped(
        root=get_dataset_path(dataset="ucsd"),
        category="UCSDped2",
        image_size=(256, 256),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        task=task,
        val_split_mode=val_split_mode,
    )
    data_module.setup()
    return data_module


def make_visa_data_module(task="classification", batch_size=1, test_split_mode="from_dir", val_split_mode="from_test"):
    data_module = Visa(
        root=get_dataset_path(dataset="visa"),
        category="candle",
        image_size=(256, 256),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        task=task,
        test_split_mode=test_split_mode,
        val_split_mode=val_split_mode,
    )
    data_module.prepare_data()
    data_module.setup()
    return data_module


DATASETS = {
    "avenue": make_avenue_data_module,
    "btech": make_btech_data_module,
    "folder": make_folder_data_module,
    "mvtec": make_mvtec_data_module,
    "shanghaitech": make_shanghaitech_data_module,
    "ucsdped": make_ucsdped_data_module,
    "visa": make_visa_data_module,
}


@pytest.fixture(autouse=True)
def make_data_module():
    def make(dataset="folder", **kwargs):
        return DATASETS[dataset](**kwargs)

    return make


@pytest.fixture(autouse=True)
def data_sample():
    root = get_dataset_path(dataset="bottle")
    datamodule = Folder(
        root=root,
        normal_dir="good",
        abnormal_dir="broken_large",
        mask_dir="ground_truth/broken_large",
        normal_split_ratio=0.2,
        image_size=(256, 256),
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        task="classification",
        val_split_mode="from_test",
    )
    datamodule.setup()
    _, data = next(enumerate(datamodule.train_dataloader()))
    return data


@pytest.mark.parametrize("dataset", ["avenue", "btech", "folder", "mvtec", "shanghaitech", "ucsdped", "visa"])
class TestDataModule:
    """Test MVTec AD Data Module."""

    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_batch_size(self, make_data_module, dataset, batch_size):
        """Test if both single and multiple batch size returns outputs with the right shape."""
        data_module = make_data_module(dataset=dataset, batch_size=batch_size)
        _, train_data_sample = next(enumerate(data_module.train_dataloader()))
        _, val_data_sample = next(enumerate(data_module.val_dataloader()))
        assert train_data_sample["image"].shape[0] == batch_size
        assert val_data_sample["image"].shape[0] == batch_size

    @pytest.mark.parametrize("task", ["classification", "detection", "segmentation"])
    def test_keys_in_batch_dict(self, make_data_module, dataset, task):
        """Test if the batch returned by __getitem__ contains the required keys."""
        data_module = make_data_module(dataset=dataset, task=task)
        _, val_data = next(enumerate(data_module.val_dataloader()))
        _, test_data = next(enumerate(data_module.test_dataloader()))

        assert set(EXPECTED_KEYS_PER_TASK_TYPE[task]).issubset(set(val_data.keys()))
        assert set(EXPECTED_KEYS_PER_TASK_TYPE[task]).issubset(set(test_data.keys()))

    def test_non_overlapping_splits(self, make_data_module, dataset):
        """This test ensures that all splits are non-overlapping when split mode == from_test."""
        data_module = make_data_module(dataset=dataset, val_split_mode="from_test")
        assert (
            len(
                set(data_module.test_data.samples["image_path"].values).intersection(
                    set(data_module.train_data.samples["image_path"].values)
                )
            )
            == 0
        ), "Found train and test split contamination"
        assert (
            len(
                set(data_module.val_data.samples["image_path"].values).intersection(
                    set(data_module.test_data.samples["image_path"].values)
                )
            )
            == 0
        ), "Found train and test split contamination"

    def test_equal_splits(self, make_data_module, dataset):
        """This test ensures that val and test split are equal when split mode == same_as_test."""
        data_module = make_data_module(dataset=dataset, val_split_mode="same_as_test")
        assert all(
            data_module.val_data.samples["image_path"].values == data_module.test_data.samples["image_path"].values
        )


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


class TestSubsetSplitting:
    @pytest.mark.parametrize("dataset", ["folder"])
    @pytest.mark.parametrize("test_split_mode", ("from_dir", "synthetic"))
    @pytest.mark.parametrize("val_split_mode", ("from_test", "synthetic"))
    def test_non_overlapping_splits(self, make_data_module, dataset, test_split_mode, val_split_mode):
        """Tests if train, test and val splits are non-overlapping."""
        data_module = make_data_module(dataset, test_split_mode=test_split_mode, val_split_mode=val_split_mode)
        train_samples = data_module.train_data.samples
        val_samples = data_module.val_data.samples
        test_samples = data_module.test_data.samples
        assert len(set(train_samples.image_path).intersection(set(test_samples.image_path))) == 0
        assert len(set(val_samples.image_path).intersection(set(test_samples.image_path))) == 0

    @pytest.mark.parametrize("dataset", ["folder"])
    @pytest.mark.parametrize("test_split_mode", ("from_dir", "synthetic"))
    def test_equal_splits(self, make_data_module, dataset, test_split_mode):
        """Tests if test and and val splits are equal and non-overlapping with train."""
        data_module = make_data_module(dataset, test_split_mode=test_split_mode, val_split_mode="same_as_test")
        train_samples = data_module.train_data.samples
        val_samples = data_module.val_data.samples
        test_samples = data_module.test_data.samples
        assert len(set(train_samples.image_path).intersection(set(test_samples.image_path))) == 0
        assert len(set(val_samples.image_path).intersection(set(test_samples.image_path))) == len(val_samples)

    @pytest.mark.parametrize("test_split_mode", ("from_dir", "synthetic"))
    def test_normal_test_dir_omitted(self, make_data_module, test_split_mode):
        """Tests if the data module functions properly when no normal_test_dir is provided."""
        data_module = make_data_module(dataset="folder", test_split_mode=test_split_mode, normal_test_dir=None)
        # check if we can retrieve a sample from every subset
        next(iter(data_module.train_dataloader()))
        next(iter(data_module.test_dataloader()))
        next(iter(data_module.val_dataloader()))
        # the test set should contain normal samples which are sampled from the train set
        assert data_module.test_data.has_normal

    def test_abnormal_dir_omitted_from_dir(self, make_data_module):
        """The test set should not contain anomalous samples if no abnormal_dir provided and split mode is from_dir."""
        data_module = make_data_module(dataset="folder", test_split_mode="from_dir", abnormal_dir=None)
        # check if we can retrieve a sample from every subset
        next(iter(data_module.train_dataloader()))
        next(iter(data_module.test_dataloader()))
        next(iter(data_module.val_dataloader()))
        # the test set should not contain anomalous samples, because there aren't any available
        assert not data_module.test_data.has_anomalous

    def test_abnormal_dir_omitted_synthetic(self, make_data_module):
        """The test set should contain anomalous samples if no abnormal_dir provided and split mode is synthetic."""
        data_module = make_data_module(dataset="folder", test_split_mode="synthetic", abnormal_dir=None)
        # check if we can retrieve a sample from every subset
        next(iter(data_module.train_dataloader()))
        next(iter(data_module.test_dataloader()))
        next(iter(data_module.val_dataloader()))
        # the test set should contain anomalous samples, which have been converted from normals
        assert data_module.test_data.has_anomalous

    def test_masks_dir_omitted(self, make_data_module):
        """Tests if the data module can be set up in classification mode when no masks are passed."""
        data_module = make_data_module(dataset="folder", task="classification", mask_dir=None)
        # check if we can retrieve a sample from every subset
        next(iter(data_module.train_dataloader()))
        next(iter(data_module.test_dataloader()))
        next(iter(data_module.val_dataloader()))
