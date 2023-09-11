"""Setup everything that should be shared between notebooks.
"""

import argparse
import json
import socket
from functools import lru_cache, wraps
from pathlib import Path
from time import time

import pandas as pd
import PIL
import torch
from pytorch_lightning import seed_everything
from skimage import morphology as skm

from anomalib.data import MVTec, TaskType, Visa
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.data.utils.image import read_image
from anomalib.post_processing import NormalizationMethod, ThresholdMethod, superimpose_anomaly_map
from anomalib.utils.callbacks import MetricsConfigurationCallback, PostProcessingConfigurationCallback
from anomalib.utils.metrics import AUPR, AUPRO, AUROC
from anomalib.utils.metrics.perimg import AULogPImO

# =============================================================================
# CONSTANTS

DATADIR = Path(__file__).parent / "data"
DATADIR.mkdir(exist_ok=True, parents=True)
DATASETSDIR = Path.home() / "data/datasets"

# DATADIR = Path("data/data-musca")
# DATADIR = Path("data/data-fon")

# DATASETSDIR = DATADIR / 'datasets'

print(f"{DATADIR=}")
print(f"{DATASETSDIR=}")

MVTECDIR = DATASETSDIR / "MVTec"
VISADIR = DATASETSDIR / "VisA"

assert DATADIR.exists()
assert MVTECDIR.exists()
assert VISADIR.exists()

HOSTNAME = socket.gethostname()
print(f"{HOSTNAME=}")

OUTPUTDIR = DATADIR / "benchmark"
OUTPUTDIR.mkdir(exist_ok=True, parents=True)
print(f"{OUTPUTDIR=}")


def get_model_dir(model: str):
    d = OUTPUTDIR / model
    d.mkdir(exist_ok=True, parents=True)
    return d


INPUT_IMAGE_RESOLUTION = 256
print(f"{INPUT_IMAGE_RESOLUTION=}")

DATASET_MVTEC = "mvtec"
DATASET_VISA = "visa"
DATASET_CHOICES = [DATASET_MVTEC, DATASET_VISA]
print(f"{DATASET_CHOICES=}")

CATEGORY_CHOICES_MVTEC = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

CATEGORY_CHOICES_VISA = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

DATASET_CATEGORY_CHOICES = [
    f"{ds}/{cat}"
    for ds, cats in [
        (DATASET_MVTEC, CATEGORY_CHOICES_MVTEC),
        (DATASET_VISA, CATEGORY_CHOICES_VISA),
    ]
    for cat in cats
]


# =============================================================================
# PARSER

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-category", "--dc", choices=DATASET_CATEGORY_CHOICES, nargs="*", required=True)
parser.add_argument("--seed-global", type=int, default=0)
parser.add_argument("--seed-datamodule", type=int, default=0)


# =============================================================================
# SEEDING


def get_global_seeder(seed):
    def global_seeder():
        torch.manual_seed(seed)
        seed_everything(seed, workers=True)

    return global_seeder


# =============================================================================
# DATA


def get_datamodule_mvtec(category: str, train_batch_size: int, eval_batch_size: int, num_workers: int, seed: int):
    assert category in CATEGORY_CHOICES_MVTEC, f"{category=}"
    return MVTec(
        root=MVTECDIR,
        category=category,
        image_size=INPUT_IMAGE_RESOLUTION,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        task=TaskType.SEGMENTATION,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        seed=seed,
    )


def get_datamodule_visa(category: str, train_batch_size: int, eval_batch_size: int, num_workers: int, seed: int):
    assert category in CATEGORY_CHOICES_VISA, f"{category=}"
    return Visa(
        root=VISADIR,
        category=category,
        image_size=INPUT_IMAGE_RESOLUTION,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        task=TaskType.SEGMENTATION,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        seed=seed,
    )


def _get_datamodule(dataset, category, **kwargs):
    assert dataset in DATASET_CHOICES, f"{dataset=}"

    if dataset == DATASET_MVTEC:
        return get_datamodule_mvtec(category, **kwargs)

    if dataset == DATASET_VISA:
        return get_datamodule_visa(category, **kwargs)

    raise NotImplementedError(f"{dataset=}")


@lru_cache
def get_datamodule(dataset, category, **kwargs):
    dm = _get_datamodule(dataset, category, **kwargs)
    dm.prepare_data()
    dm.setup()
    return _get_datamodule(dataset, category, **kwargs)


def get_test_img_name(img_relpath: str, dataset: str, category: str) -> str:
    assert dataset in DATASET_CHOICES, f"{dataset=}"

    if dataset == DATASET_MVTEC:
        test_img_dir = MVTECDIR / category / "test"

    elif dataset == DATASET_VISA:
        test_img_dir = VISADIR / f"visa_pytorch/{category}/test"

    else:
        raise NotImplementedError(f"{dataset=}")

    return str(Path(img_relpath).relative_to(test_img_dir))


# =============================================================================
# CALLBACKS
# They are the same for all models and don't matter much but the trainer needs them.

METRICS_CALLBACK = MetricsConfigurationCallback(
    task=TaskType.SEGMENTATION,
    # these are empty becasue they will be computed manually so they
    # can be more easily timed
    image_metrics=[],
    pixel_metrics=[],
)

POSTPROCESSING_CALLBACK = PostProcessingConfigurationCallback(
    normalization_method=NormalizationMethod.MIN_MAX,
    threshold_method=ThresholdMethod.ADAPTIVE,
)

STANDARD_CALLBACKS = [METRICS_CALLBACK, POSTPROCESSING_CALLBACK]


# =============================================================================
# TRAIN


def train(dataset, category, datamodule, model, trainer, savedir):
    trainer.fit(datamodule=datamodule, model=model)
    torch.save(model.state_dict(), savedir / "model_state_dict.pt")  # DEBUG


# =============================================================================
# TEST


def test(dataset, category, datamodule, model, trainer, savedir):
    # `asmap` stands for `Anomaly Score MAP`
    # `ascore` stands for `Anomaly Score`
    predictions = trainer.predict(model=model, dataloaders=datamodule.test_dataloader())

    asmaps = torch.concatenate(
        [batch["anomaly_maps"][idx].cpu() for batch in predictions for idx in range(len(batch["image"]))], dim=0
    )

    masks = torch.stack(
        [batch["mask"][idx].cpu() for batch in predictions for idx in range(len(batch["image"]))], dim=0
    ).int()

    # `tmp` is just to make the variable easier to read
    df = pd.DataFrame.from_records(
        [
            {
                "imgpath": (imgpath := batch["image_path"][idx]),
                "maskpath": maskpath if len(maskpath := batch["mask_path"][idx]) > 0 else pd.NA,
                # ex (mvtec): "broken_large/000.png"
                "imgname": get_test_img_name(imgpath, dataset, category),
                "ascore": batch["pred_scores"][idx].item(),
            }
            for batch in predictions
            for idx in range(len(batch["image"]))
        ]
    )
    df = df.astype({"imgpath": "string", "maskpath": "string", "imgname": "string"})
    df["imgclass"] = (~df["maskpath"].isna()).astype(int)

    # sort by image name (label/file name)
    # and make sure `asmaps` is sorted like the dataframe
    df.reset_index(inplace=True)  # create the `index` column
    df.sort_values(by="imgname", inplace=True)
    asmaps = asmaps[df["index"].values]
    df = df.drop(columns=["index"]).reset_index(drop=False)

    df.to_csv(savedir / "predictions.csv", index=False)
    torch.save(asmaps, savedir / "asmaps.pt")

    return df, asmaps, masks


# =============================================================================
# EVAL


def get_contour_mask(msk):
    return skm.binary_dilation(~msk, skm.square(3)) * msk


def evaluate(imgpaths, ascores, imgclass, asmaps, masks, savedir, logger, debug):
    times = {}

    def logtime(f):
        @wraps(f)
        def wrap(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()
            sec = te - ts
            record = {f"{f.__name__}_sec": sec}
            logger.experiment.log(record)
            times.update(record)
            return result

        return wrap

    if debug:
        # makes it faster to debug
        some_norm = torch.where(imgclass == 0)[0][:2]
        some_anom = torch.where(imgclass == 1)[0][:2]
        some_imgs = torch.cat([some_norm, some_anom])
        asmaps = asmaps[some_imgs]
        masks = masks[some_imgs]

    img_auroc = AUROC()
    img_auroc.update(ascores, imgclass)
    img_auroc = img_auroc.compute().item()

    img_aupr = AUPR()
    img_aupr.update(ascores, imgclass)
    img_aupr = img_aupr.compute().item()

    img_metrics = {
        "img_auroc": img_auroc,
        "img_aupr": img_aupr,
    }
    pd.DataFrame.from_records([img_metrics]).to_csv(savedir / "img_metrics.csv", index=False)
    logger.experiment.log(img_metrics)

    @logtime
    def get_pix_auroc(asmaps, masks):
        pix_auroc = AUROC()
        pix_auroc.update(asmaps, masks)
        return pix_auroc.compute().item()

    pix_auroc = get_pix_auroc(asmaps, masks)

    @logtime
    def get_pix_aupr(asmaps, masks):
        pix_aupr = AUPR()
        pix_aupr.update(asmaps, masks)
        return pix_aupr.compute().item()

    pix_aupr = get_pix_aupr(asmaps, masks)

    @logtime
    def get_aupro(asmaps, masks):
        aupro = AUPRO()
        aupro.update(asmaps, masks)
        return aupro.compute().item()

    # aupro = get_aupro(asmaps[:1], masks[:1])
    aupro = get_aupro(asmaps, masks)

    pix_set_metrics = {
        "pix_set_auroc": pix_auroc,
        "pix_set_aupr": pix_aupr,
        "pix_set_aupro": aupro,
    }
    logger.experiment.log(pix_set_metrics)
    pd.DataFrame.from_records([pix_set_metrics]).to_csv(savedir / "pix_set_metrics.csv", index=False)

    @logtime
    def get_aulogpimo(asmaps, masks):
        aulogpimo = AULogPImO(lbound=0.0001, ubound=0.01)
        aulogpimo.cpu()
        aulogpimo.update(asmaps, masks)
        return aulogpimo, aulogpimo.compute()

    aulogpimo, (pimoresult, aucresult) = get_aulogpimo(asmaps, masks)
    boxplot_stats = sorted(aulogpimo.boxplot_stats(), key=lambda x: x["value"])

    aulogpimo_dir = savedir / "aulogpimo_001_1"
    aulogpimo_dir.mkdir(exist_ok=True, parents=True)

    pimoresult.save(aulogpimo_dir / "curves.pt")
    aucresult.save(aulogpimo_dir / "aucs.json")

    with (aulogpimo_dir / "boxplot_stats.json").open("w") as f:
        json.dump(boxplot_stats, f, indent=4)

    th_normalization = (aucresult.ubound_threshold.item(), aucresult.lbound_threshold.item())

    # save some vizualizations
    for idx, stat in enumerate(boxplot_stats):
        imgidx = stat["imgidx"]

        filename = f"{idx:03}_{stat['statistic']}_imgidx={imgidx}_auc={stat['value']:.4f}.png"

        imgpath = imgpaths[imgidx]
        img = read_image(imgpath, image_size=(INPUT_IMAGE_RESOLUTION, INPUT_IMAGE_RESOLUTION))

        asmap = asmaps[imgidx].cpu().numpy()
        supimg = superimpose_anomaly_map(asmap, img, normalize=th_normalization, alpha=0.3)

        mask = masks[imgidx].cpu().numpy()
        mask_contour = get_contour_mask(mask.astype(bool))
        supimg = superimpose_anomaly_map(mask_contour.astype(float), supimg, normalize=(0.98, 0.99), alpha=1)

        PIL.Image.fromarray(supimg).save(aulogpimo_dir / filename)

    with (savedir / "times.json").open("w") as f:
        json.dump(times, f, indent=4)
