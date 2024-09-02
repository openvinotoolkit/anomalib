"""Test Helpers - Dataset."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from contextlib import ContextDecorator
from pathlib import Path
from tempfile import mkdtemp

import cv2
import numpy as np
from scipy.io import savemat
from skimage import img_as_ubyte
from skimage.io import imsave

from anomalib.data import DataFormat
from anomalib.data.utils import Augmenter, LabelName


class DummyImageGenerator:
    """Dummy image generator.

    Args:
        image_shape (tuple[int, int], optional): Image shape. Defaults to (256, 256).

    Examples:
        To generate a normal image, use the ``generate_normal_image`` method.
        >>> generator = DummyImageGenerator()
        >>> image, mask = generator.generate_normal_image()

        To generate an abnormal image, use the ``generate_abnormal_image`` method.
        >>> generator = DummyImageGenerator()
        >>> image, mask = generator.generate_abnormal_image()

        To generate an image with a specific label, use the ``generate_image`` method.
        >>> generator = DummyImageGenerator()
        >>> image, mask = generator.generate_image(label=LabelName.ABNORMAL)

        or,
        >>> generator = DummyImageGenerator()
        >>> image, mask = generator.generate_image(label=LabelName.NORMAL)
    """

    def __init__(self, image_shape: tuple[int, int] = (256, 256), rng: np.random.Generator | None = None) -> None:
        self.image_shape = image_shape
        self.augmenter = Augmenter()
        self.rng = rng if rng else np.random.default_rng()

    def generate_normal_image(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a normal image."""
        image = np.zeros([*self.image_shape, 3]).astype(np.uint8)
        image[...] = self.rng.integers(low=0, high=255, size=[3])
        mask = np.zeros(self.image_shape).astype(np.uint8)

        return image, mask

    def generate_abnormal_image(self, beta: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """Generate an abnormal image.

        Args:
            beta (float, optional): beta value for superimposing perturbation on image. Defaults to 0.2.

        Returns:
            tuple[np.ndarray, np.ndarray]: Abnormal image and mask.
        """
        # Generate a random image.
        image, _ = self.generate_normal_image()

        # Generate perturbation.
        perturbation, mask = self.augmenter.generate_perturbation(height=self.image_shape[0], width=self.image_shape[1])

        # Superimpose perturbation on image ``img``.
        abnormal_image = (image * (1 - mask) + (beta) * perturbation + (1 - beta) * image * (mask)).astype(np.uint8)

        return abnormal_image, mask.squeeze()

    def generate_image(
        self,
        label: LabelName = LabelName.NORMAL,
        image_filename: Path | str | None = None,
        mask_filename: Path | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a random image.

        Args:
            label (LabelName, optional): Image label (NORMAL - 0 or ABNORMAL - 1). Defaults to 0.
            image_filename (Path | str | None, optional): Image filename to save to filesytem. Defaults to None.
            mask_filename (Path | str | None, optional): Mask filename to save to filesystem. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: Image and mask.
        """
        func = self.generate_normal_image if label == LabelName.NORMAL else self.generate_abnormal_image
        image, mask = func()

        if image_filename:
            self.save_image(filename=image_filename, image=image)

        if mask_filename:
            self.save_image(filename=mask_filename, image=img_as_ubyte(mask))

        return image, mask

    @staticmethod
    def save_image(filename: Path | str, image: np.ndarray, check_contrast: bool = False) -> None:
        """Save image to filesystem.

        Args:
            filename (Path | str): Filename to save image to.
            image (np.ndarray): Image to save.
            check_contrast (bool, optional): Check for low contrast and print warning. Defaults to False.
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        imsave(fname=filename, arr=image, check_contrast=check_contrast)


class DummyVideoGenerator(DummyImageGenerator):
    """Dummy video generator.

    Args:
        num_frames (int, optional): Length of the video. Defaults to 32.
        frame_shape (tuple[int, int], optional): Shape of the video frames. Defaults to (256, 256).

    Examples:
        To generate a video with a random sequence of normal and abnormal frames, use the ``generate_video`` method.
        >>> generator = DummyVideoGenerator()
        >>> frames, masks = generator.generate_video()

        It is possible to specify the length of the video
        >>> generator = DummyVideoGenerator()
        >>> frames, masks = generator.generate_video(length=64)

        It is possible to specify the first frame label (normal or abnormal)
        >>> generator = DummyVideoGenerator()
        >>> frames, masks = generator.generate_video(first_label=LabelName.ABNORMAL)

        It is possible to specify the probability of switching between normal and abnormal frames
        >>> generator = DummyVideoGenerator()
        >>> frames, masks = generator.generate_video(p_state_switch=0.5)
    """

    def __init__(self, num_frames: int = 32, frame_shape: tuple[int, int] = (256, 256)) -> None:
        super().__init__(frame_shape)
        self.num_frames = num_frames

    def generate_video(
        self,
        length: int = 32,
        first_label: LabelName = LabelName.NORMAL,
        p_state_switch: float = 0.2,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Generate video clip with a random sequence of anomalous frames.

        Args:
            length (int): Length of the video sequence in number of frames.
            first_label (LabelName): Label of the first frame (normal or abnormal).
            p_state_switch (float): Probability of transitioning between normal and anomalous in consecutive frames.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: List of frames and list of masks.
        """
        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        state = 1 if first_label == LabelName.NORMAL else -1
        for _ in range(length):
            state = state * -1 if self.rng.random() < p_state_switch else state
            label = LabelName.NORMAL if state == 1 else LabelName.ABNORMAL
            frame, mask = self.generate_image(label=label)
            frames.append(frame)
            masks.append(mask)
        return frames, masks

    def save_frame(self, filename: Path | str, frame: np.ndarray, check_contrast: bool = False) -> None:
        """Save frame to filesystem.

        Args:
            filename (Path | str): Filename to save image to.
            frame (np.ndarray): Frame to save.
            check_contrast (bool, optional): Check for low contrast and print warning. Defaults to False.
        """
        self.save_image(filename=filename, image=frame, check_contrast=check_contrast)


class DummyDatasetGenerator(ContextDecorator):
    """Base dummy dataset generator class to be implemented by Image and Video generators.

    Args:
        data_format (DataFormat): Data format of the dataset.
        root (Path | str | None, optional): Root directory to save the dataset. Defaults to None.
        dataset_name (str, optional): Name of the dataset. Defaults to None.
        num_train (int, optional): Number of training images to generate. Defaults to 5.
        num_test (int, optional): Number of testing images to generate per category. Defaults to 5.
        seed (int, optional): Fixes seed if any number greater than 0 is provided. 0 means no seed. Defaults to 0.

    Examples:
        >>> with DummyDatasetGenerator(num_train=10, num_test=10) as dataset_path:
        >>>     some_function()

        Alternatively, you can use it as a standalone class.
        This will create a temporary directory with the dataset.

        >>> generator = DummyDatasetGenerator(num_train=10, num_test=10)
        >>> generator.generate_dataset()

        If you want to use a specific directory, you can pass it as an argument.

        >>> generator = DummyDatasetGenerator(root="./datasets/dummy")
        >>> generator.generate_dataset()
    """

    def __init__(
        self,
        data_format: DataFormat | str,
        root: Path | str | None = None,
        num_train: int = 5,
        num_test: int = 5,
        seed: int | None = None,
    ) -> None:
        if isinstance(data_format, str):
            data_format = DataFormat(data_format)

        if data_format not in list(DataFormat):
            message = f"Invalid data format {data_format}. Valid options are {list(DataFormat)}."
            raise ValueError(message)

        self.data_format = data_format
        self.root = Path(mkdtemp() if root is None else root)
        self.dataset_root = self.root / self.data_format.value
        self.num_train = num_train
        self.num_test = num_test
        self.rng = np.random.default_rng(seed)

    def generate_dataset(self) -> None:
        """Generate dataset."""
        # get dataset specific ``generate_dataset`` function based on string.

        if hasattr(self, f"_generate_dummy_{self.data_format.value}_dataset"):
            method_name = f"_generate_dummy_{self.data_format.value}_dataset"
            method = getattr(self, method_name)
            method()
        else:
            message = f"``generate_dummy_{self.data_format.value}_dataset`` not implemented."
            raise NotImplementedError(message)

    def __enter__(self) -> Path:
        """Creates the dataset in temp folder."""
        self.generate_dataset()
        return self.dataset_root

    def __exit__(self, _exc_type, _exc_value, _exc_traceback) -> None:  # noqa: ANN001
        """Cleanup the directory."""
        shutil.rmtree(self.dataset_root)


class DummyImageDatasetGenerator(DummyDatasetGenerator):
    r"""Context for generating dummy shapes dataset.

    Args:
        data_format (DataFormat): Data format of the dataset.
        root (Path | str, optional): Path to the root directory. Defaults to None.
        num_train (int, optional): Number of training images to generate. Defaults to 1000.
        num_test (int, optional): Number of testing images to generate per category. Defaults to 100.
        img_height (int, optional): Height of the image. Defaults to 128.
        img_width (int, optional): Width of the image. Defaults to 128.
        max_size (Optional[int], optional): Maximum size of the test shapes. Defaults to 10.
        train_shapes (List[str], optional): List of good shapes. Defaults to ["circle", "rectangle"].
        test_shapes (List[str], optional): List of anomalous shapes. Defaults to ["triangle", "ellipse"].
        seed (int, optional): Fixes seed if any number greater than 0 is provided. 0 means no seed. Defaults to 0.

    Examples:
        To create an MVTec dataset with 10 training images and 10 testing images per category, use the following code.
        >>> dataset_generator = DummyImageDatasetGenerator(data_format="mvtec", num_train=10, num_test=10)
        >>> dataset_generator.generate_dataset()

        In order to provide a specific directory to save the dataset, use the ``root`` argument.
        >>> dataset_generator = DummyImageDatasetGenerator(data_format="mvtec", root="./datasets/dummy")
        >>> dataset_generator.generate_dataset()

        It is also possible to use the generator as a context manager.
        >>> with DummyImageDatasetGenerator(data_format="mvtec", num_train=10, num_test=10) as dataset_path:
        >>>     some_function()

        To get the list of available datasets, use the ``DataFormat`` enum.
        >>> from anomalib.data import DataFormat
        >>> print(list(DataFormat))

        Then you can use the ``DataFormat`` enum to generate the dataset.
        >>> dataset_generator = DummyImageDatasetGenerator(data_format="beantech", num_train=10, num_test=10)
    """

    def __init__(
        self,
        data_format: DataFormat | str = "mvtec",
        root: Path | str | None = None,
        normal_category: str = "good",
        abnormal_category: str = "bad",
        num_train: int = 5,
        num_test: int = 5,
        image_shape: tuple[int, int] = (256, 256),
        num_channels: int = 3,
        min_size: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            data_format=data_format,
            root=root,
            num_train=num_train,
            num_test=num_test,
            seed=seed,
        )
        self.normal_category = normal_category
        self.abnormal_category = abnormal_category
        self.image_shape = image_shape
        self.num_channels = num_channels
        self.min_size = min_size
        self.image_generator = DummyImageGenerator(image_shape=image_shape, rng=self.rng)

    def _generate_dummy_mvtec_dataset(
        self,
        normal_dir: str = "good",
        abnormal_dir: str | None = None,
        image_extension: str = ".png",
        mask_suffix: str = "_mask",
        mask_extension: str = ".png",
    ) -> None:
        """Generates dummy MVTecAD dataset in a temporary directory using the same convention as MVTec AD."""
        # MVTec has multiple subcategories within the dataset.
        dataset_category = "dummy"

        # Create normal images.
        for split in ("train", "test"):
            path = self.dataset_root / dataset_category / split / normal_dir
            num_images = self.num_train if split == "train" else self.num_test
            for i in range(num_images):
                label = LabelName.NORMAL
                image_filename = path / f"{i:03}{image_extension}"
                self.image_generator.generate_image(label=label, image_filename=image_filename)

        # Create abnormal test images and masks.
        abnormal_dir = abnormal_dir or self.abnormal_category
        path = self.dataset_root / dataset_category / "test" / abnormal_dir
        mask_path = self.dataset_root / dataset_category / "ground_truth" / abnormal_dir

        for i in range(self.num_test):
            label = LabelName.ABNORMAL
            image_filename = path / f"{i:03}{image_extension}"
            mask_filename = mask_path / f"{i:03}{mask_suffix}{mask_extension}"
            self.image_generator.generate_image(label, image_filename, mask_filename)

    def _generate_dummy_btech_dataset(self) -> None:
        """Generate dummy BeanTech dataset in directory using the same convention as BeanTech AD."""
        # BeanTech AD follows the same convention as MVTec AD.
        self._generate_dummy_mvtec_dataset(normal_dir="ok", abnormal_dir="ko", mask_suffix="")

    def _generate_dummy_mvtec_3d_dataset(self) -> None:
        """Generate dummy MVTec 3D AD dataset in a temporary directory using the same convention as MVTec AD."""
        # MVTec 3D AD has multiple subcategories within the dataset.
        dataset_category = "dummy"

        # Create training and validation images.
        for split in ("train", "validation"):
            split_path = self.dataset_root / dataset_category / split / self.normal_category
            for directory in ("rgb", "xyz"):
                extension = ".tiff" if directory == "xyz" else ".png"
                for i in range(self.num_train):
                    label = LabelName.NORMAL
                    image_filename = split_path / directory / f"{i:03}{extension}"
                    self.image_generator.generate_image(label=label, image_filename=image_filename)

        ## Create test images.
        test_path = self.dataset_root / dataset_category / "test"
        for category in (self.normal_category, self.abnormal_category):
            label = LabelName.NORMAL if category == "good" else LabelName.ABNORMAL
            for i in range(self.num_test):
                # Generate image and mask.
                image, mask = self.image_generator.generate_image(label=label)

                # Create rgb, xyz, and gt filenames.
                for directory in ("rgb", "xyz", "gt"):
                    extension = ".png" if directory == "gt" else ".tiff" if directory == "xyz" else ".png"
                    filename = test_path / category / directory / f"{i:03}{extension}"

                    # Save image or mask.
                    if directory == "gt":
                        self.image_generator.save_image(filename=filename, image=img_as_ubyte(mask))
                    else:
                        self.image_generator.save_image(filename=filename, image=image)

    def _generate_dummy_kolektor_dataset(self) -> None:
        """Generate dummy Kolektor dataset in directory using the same convention as Kolektor AD."""
        # Emulating the first two categories of Kolektor dataset.
        for category in ("kos01", "kos02"):
            for i in range(self.num_train * 2):
                # Half of the images are normal, while the rest are abnormal.
                label = LabelName.NORMAL if i > self.num_train // 2 else LabelName.ABNORMAL
                image_filename = self.dataset_root / category / f"Part{i}.jpg"
                mask_filename = self.dataset_root / category / f"Part{i}_label.bmp"
                self.image_generator.generate_image(label, image_filename, mask_filename)

    def _generate_dummy_visa_dataset(self) -> None:
        """Generate dummy Visa dataset in directory using the same convention as Visa AD."""
        # Visa dataset on anomalib follows the same convention as MVTec AD.
        # The only difference is that the root directory has a subdirectory called "visa_pytorch".
        self.dataset_root = self.dataset_root.parent / "visa_pytorch"
        self._generate_dummy_mvtec_dataset(normal_dir="good", abnormal_dir="bad", image_extension=".jpg")


class DummyVideoDatasetGenerator(DummyDatasetGenerator):
    """Dummy video dataset generator.

    Args:
        data_format (DataFormat): Data format of the dataset.
        root (Path | str | None, optional): Root directory to save the dataset. Defaults to None.
        dataset_name (str, optional): Name of the dataset. Defaults to "ucsdped1".
        num_frames (int, optional): Number of frames to generate the video. Defaults to 32.
        frame_shape (tuple[int, int], optional): Shape of individual frames. Defaults to (256, 256).
        num_train (int, optional): Number of training images to generate. Defaults to 5.
        num_test (int, optional): Number of testing images to generate per category. Defaults to 5.
        seed (int, optional): Fixes seed if any number greater than 0 is provided. 0 means no seed. Defaults to 0.

    Examples:
        To create a UCSDped1 dataset with 10 training videos and 10 testing videos, use the following code.
        >>> dataset_generator = DummyVideoDatasetGenerator(data_format="ucsdped", num_train=10, num_test=10)
        >>> dataset_generator.generate_dataset()

        In order to provide a specific directory to save the dataset, use the ``root`` argument.
        >>> dataset_generator = DummyVideoDatasetGenerator(data_format="ucsdped", root="./datasets/dummy")
        >>> dataset_generator.generate_dataset()

        It is also possible to use the generator as a context manager.
        >>> with DummyVideoDatasetGenerator(data_format="ucsdped", num_train=10, num_test=10) as dataset_path:
        >>>     some_function()

        To get the list of available datasets, use the ``VideoDataFormat`` enum.
        >>> from anomalib.data import VideoDataFormat
        >>> print(list(VideoDataFormat))

        Based on the enum, you can generate the dataset.
        >>> dataset_generator = DummyVideoDatasetGenerator(data_format="avenue", num_train=10, num_test=10)
        >>> dataset_generator.generate_dataset()
    """

    def __init__(
        self,
        data_format: DataFormat,
        root: Path | str | None = None,
        num_frames: int = 32,
        frame_shape: tuple[int, int] = (256, 256),
        num_train: int = 5,
        num_test: int = 5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            data_format=data_format,
            root=root,
            num_train=num_train,
            num_test=num_test,
            seed=seed,
        )
        self.video_length = num_frames
        self.frame_shape = frame_shape
        self.video_generator = DummyVideoGenerator(num_frames=num_frames, frame_shape=frame_shape)

    def _generate_dummy_ucsdped_dataset(self, train_dir: str = "Train", test_dir: str = "Test") -> None:
        """Generate dummy UCSD dataset."""
        # generate training data
        dataset_category = "dummy"
        train_path = self.dataset_root / dataset_category / train_dir
        for clip_idx in range(self.num_train):
            clip_name = train_path / f"Train{clip_idx:03}"
            frames, _ = self.video_generator.generate_video(
                length=self.video_length,
                first_label=LabelName.NORMAL,
                p_state_switch=0,
            )
            for frame_idx, frame in enumerate(frames):
                filename = clip_name / f"{frame_idx:03}.tif"
                self.video_generator.save_frame(filename, frame)

        # generate test data
        test_path = self.dataset_root / dataset_category / test_dir
        for clip_idx in range(self.num_test):
            clip_path = test_path / f"Test{clip_idx:03}"
            mask_path = test_path / f"Test{clip_idx:03}_gt"
            frames, masks = self.video_generator.generate_video(length=self.video_length, p_state_switch=0.2)
            for frame_idx, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
                filename_frame = clip_path / f"{frame_idx:03}.tif"
                filename_mask = mask_path / f"{frame_idx:03}.bmp"
                self.video_generator.save_frame(filename_frame, frame)
                self.video_generator.save_frame(filename_mask, (mask * 255).astype(np.uint8))

    def _generate_dummy_avenue_dataset(
        self,
        train_dir: str = "training_videos",
        test_dir: str = "testing_videos",
        ground_truth_dir: str = "ground_truth_demo",
    ) -> None:
        """Generate dummy Avenue dataset."""
        # generate training data
        train_path = self.dataset_root / train_dir
        train_path.mkdir(exist_ok=True, parents=True)
        for clip_idx in range(self.num_train):
            clip_path = train_path / f"{clip_idx + 1:02}.avi"
            frames, _ = self.video_generator.generate_video(length=32, first_label=LabelName.NORMAL, p_state_switch=0)
            fourcc = cv2.VideoWriter_fourcc("F", "M", "P", "4")
            writer = cv2.VideoWriter(str(clip_path), fourcc, 30, self.frame_shape)
            for _, frame in enumerate(frames):
                writer.write(frame)
            writer.release()

        # generate test data
        test_path = self.dataset_root / test_dir
        test_path.mkdir(exist_ok=True, parents=True)
        gt_path = self.dataset_root / ground_truth_dir / "testing_label_mask"

        for clip_idx in range(self.num_test):
            clip_path = test_path / f"{clip_idx + 1:02}.avi"
            mask_path = gt_path / f"{clip_idx + 1}_label"
            mask_path.mkdir(exist_ok=True, parents=True)
            frames, masks = self.video_generator.generate_video(length=32, p_state_switch=0.2)
            fourcc = cv2.VideoWriter_fourcc("F", "M", "P", "4")
            writer = cv2.VideoWriter(str(clip_path), fourcc, 30, self.frame_shape)
            for frame_idx, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
                writer.write(frame)
                mask_filename = mask_path / f"{frame_idx:04}.png"
                self.video_generator.save_image(mask_filename, (mask).astype(np.uint8))
            masks_array = np.stack(masks)
            mat_filename = mask_path.with_suffix(".mat")
            savemat(mat_filename, {"data": masks_array})

    def _generate_dummy_shanghaitech_dataset(
        self,
        train_dir: str = "training",
        test_dir: str = "testing",
    ) -> None:
        """Generate dummy ShanghaiTech dataset."""
        # generate training data
        path = self.dataset_root / train_dir / "converted_videos"
        path.mkdir(exist_ok=True, parents=True)
        num_clips = self.num_train
        for clip_idx in range(num_clips):
            clip_path = path / f"01_{clip_idx:03}.avi"
            frames, _ = self.video_generator.generate_video(length=32, first_label=LabelName.NORMAL, p_state_switch=0)
            fourcc = cv2.VideoWriter_fourcc("F", "M", "P", "4")
            writer = cv2.VideoWriter(str(clip_path), fourcc, 30, self.frame_shape)
            for _, frame in enumerate(frames):
                writer.write(frame)
            writer.release()

        # generate test data
        test_path = self.dataset_root / test_dir / "frames"
        test_path.mkdir(exist_ok=True, parents=True)
        gt_path = self.dataset_root / test_dir / "test_pixel_mask"
        gt_path.mkdir(exist_ok=True, parents=True)

        for clip_idx in range(self.num_test):
            clip_path = test_path / f"01_{clip_idx:04}"
            clip_path.mkdir(exist_ok=True, parents=True)
            mask_path = gt_path / f"01_{clip_idx:04}.npy"
            frames, masks = self.video_generator.generate_video(length=32, p_state_switch=0.2)
            for frame_idx, frame in enumerate(frames):
                image_filename = clip_path / f"{frame_idx:03}.jpg"
                self.video_generator.save_image(image_filename, frame)
            masks_array = np.stack(masks)
            np.save(mask_path, masks_array)
