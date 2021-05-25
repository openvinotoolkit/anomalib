import logging
import tarfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve

import torch
from PIL import Image
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets.folder import VisionDataset, default_loader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import torchvision.transforms as T

logger = logging.getLogger(name="Dataset: MVTec")

__all__ = ["MVTec", "MVTecDataModule"]


def get_image_transforms() -> Compose:
    transform = Compose(
        [
            # ToTensor(),
            # Resize((256, 256)),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize(256, Image.ANTIALIAS),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


def get_mask_transforms() -> Compose:
    transform = Compose(
        [
            # ToTensor(),
            # Resize((256, 256)),
            T.Resize(256, Image.NEAREST), T.CenterCrop(224), T.ToTensor()
        ]
    )
    return transform


class MVTec(VisionDataset):
    def __init__(
        self,
        root: Union[Path, str],
        category: str,
        train: bool = True,
        image_transforms: Optional[Callable] = None,
        mask_transforms: Optional[Callable] = None,
        loader: Optional[Callable[[str], Any]] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=image_transforms, target_transform=mask_transforms)
        self.root = Path(root) if isinstance(root, str) else root
        self.category: Path = self.root / category
        self.split = "train" if train else "test"
        self.image_transforms = image_transforms if image_transforms is not None else get_image_transforms()
        self.mask_transforms = mask_transforms if mask_transforms is not None else get_mask_transforms()
        self.loader = loader if loader is not None else default_loader
        self.download = download

        if self.download:
            self._download()

        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 images in {self.category / self.split}")

    def _download(self) -> None:
        if self.category.is_dir():
            logger.warning("Dataset directory exists.")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self.filename = self.root / "mvtec_anomaly_detection.tar.xz"

            logger.info("Downloading MVTec Dataset")
            urlretrieve(
                url="ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz",
                filename=self.filename,
            )

            self._extract()
            self._clean()

    def _extract(self) -> None:
        logger.info("Extracting MVTec dataset")
        with tarfile.open(self.filename) as f:
            f.extractall(self.root)

    def _clean(self) -> None:
        logger.info("Cleaning up the tar file")
        self.filename.unlink()

    def make_dataset(self) -> List[Tuple[str, str, int]]:
        labels = sorted([label.name for label in (self.category / self.split).iterdir() if label.is_dir()])
        samples = []
        label_index: int

        for label in labels:
            image_filenames = sorted([filename for filename in (self.category / self.split / label).glob("**/*.png")])
            for image_filename in image_filenames:
                if label == "good":
                    label_index = 0
                    mask_filename = ""
                else:
                    label_index = 1
                    mask_filename = self.category / "ground_truth" / label / (image_filename.stem + "_mask.png")

                sample = (str(image_filename), str(mask_filename), label_index)
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]:
        image_path, mask_path, label_index = self.samples[index]

        # image = self.loader(image_path)
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transforms(image)

        if self.split == "train":
            sample = {"image": image_tensor}
        else:
            if label_index == 0:
                mask = Image.new(mode="1", size=image.size)
            else:
                mask = Image.open(mask_path).convert(mode="1")
                # mask = self.loader(mask_path)
                # mask = mask.convert(mode="1")
            # mask = self.loader(mask_path) if class_index == 1 else Image.new(mode="1", size=image.size)
            mask_tensor = self.mask_transforms(mask).to(torch.uint8)
        # else:
        #     if mask is None:
        #         # Create empty mask for good test samples.
        #         mask = Image.new(mode="1", size=image.size)
        #
        #     mask = mask.convert(mode="1")
        #     mask_tensor = self.mask_transforms(mask).to(torch.uint8)
            # sample = (image_tensor, mask_tensor)
            sample = {
                "image_path": image_path,
                "mask_path": mask_path,
                "image": image_tensor,
                "label": label_index,
                "mask": mask_tensor,
            }

        return sample


class MVTecDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        batch_size: int,
        num_workers: int,
        include_normal_images_in_val_set: bool = False,
        image_transforms: Optional[Callable] = None,
        mask_transforms: Optional[Callable] = None,
        loader: Optional[Callable[[str], Any]] = None,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path if isinstance(dataset_path, Path) else Path(dataset_path)
        self.root = self.dataset_path.parent
        self.category = self.dataset_path.stem
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.loader = default_loader if loader is None else loader
        self.include_normal = include_normal_images_in_val_set

    def prepare_data(self):
        # Training Data
        MVTec(
            root=self.root,
            category=self.category,
            train=True,
            image_transforms=self.image_transforms,
            mask_transforms=self.mask_transforms,
            download=True,
        )
        # Test Data
        MVTec(
            root=self.root,
            category=self.category,
            train=False,
            image_transforms=self.image_transforms,
            mask_transforms=self.mask_transforms,
            download=True,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_data = MVTec(
            root=self.root,
            category=self.category,
            train=True,
            image_transforms=self.image_transforms,
            mask_transforms=self.mask_transforms,
        )
        self.val_data = MVTec(
            root=self.root,
            category=self.category,
            train=False,
            # include_normal=self.include_normal,
            image_transforms=self.image_transforms,
            mask_transforms=self.mask_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        # TODO: Handle batch_size > 1
        return DataLoader(self.val_data, shuffle=False, batch_size=1, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        # TODO: Handle batch_size > 1
        return DataLoader(self.val_data, shuffle=False, batch_size=32, num_workers=self.num_workers)
