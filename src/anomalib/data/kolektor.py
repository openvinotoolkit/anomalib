from __future__ import annotations

from pathlib import Path
from pandas import DataFrame
from imageio import imread
from sklearn.model_selection import train_test_split
from anomalib.data.utils import Split
import numpy as np


def isGood(path):
    img_arr = imread(path)
    if np.all(img_arr == 0):
        return True
    else:
        return False


def make_kolektor_dataset(
    root: str | Path,
    split: str | Split | None = None,
) -> DataFrame:

    root = Path(root)
    samples_list = [
        (str(root),) + f.parts[-2:] for f in root.glob(r"**/*") if f.suffix == ".jpg"
    ]

    masks_list = [
        (str(root),) + f.parts[-2:] for f in root.glob(r"**/*") if f.suffix == ".bmp"
    ]

    if not samples_list:
        raise RuntimeError(f"Found 0 images in {root}")

    samples = DataFrame(samples_list, columns=["path", "item", "image_path"])
    masks = DataFrame(masks_list, columns=["path", "item", "image_path"])

    samples["image_path"] = samples.path + "/" + samples.item + "/" + samples.image_path
    masks["image_path"] = masks.path + "/" + masks.item + "/" + masks.image_path

    samples = samples.sort_values(by="image_path", ignore_index=True)
    masks = masks.sort_values(by="image_path", ignore_index=True)

    samples["mask_path"] = masks.image_path.values

    samples["label"] = samples["mask_path"].apply(isGood)
    samples.loc[(samples.label), "label"] = "Good"
    samples.loc[(samples.label == False), "label"] = "Bad"

    samples.loc[(samples.label == "Good"), "label_index"] = 0
    samples.loc[(samples.label == "Bad"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    # samples = samples.sort_values(by=["label", "image_path"], ignore_index=True)

    samples.loc[(samples.label == "Bad"), "split"] = "test"

    train_samples, test_samples = train_test_split(
        samples[samples.label == "Good"], train_size=0.8, test_size=0.2, random_state=42
    )

    samples.loc[train_samples.index, "split"] = "train"
    samples.loc[test_samples.index, "split"] = "test"

    samples = samples[
        ["path", "item", "split", "label", "image_path", "mask_path", "label_index"]
    ]

    # assert that the right mask files are associated with the right test images
    assert (
        samples.loc[samples.label_index == 1]
        .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
        .all()
    ), "Mismatch between anomalous images and ground truth masks. Make sure the mask files in 'ground_truth' \
              folder follow the same naming convention as the anomalous images in the dataset (e.g. image: '000.png', \
              mask: '000.png' or '000_mask.png')."

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples
