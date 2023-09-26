#!/usr/bin/env python
# coding: utf-8

# EfficientAD with WideResNet101 backbone and Student-Teacher architecture S

# In[]:
# Setup (pre-args)

from __future__ import annotations

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def get_hostname():
    import socket

    return socket.gethostname()


if "node" not in get_hostname():
    print("not in a node, setting CUDA_VISIBLE_DEVICES=2")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # DEBUG

os.environ["WANDB_NOTEBOOK_NAME"] = __file__


def is_script():
    """Returns True if running as a script (not as a notebook)."""
    import pathlib
    import sys

    arg0 = pathlib.Path(sys.argv[0]).stem
    if arg0 == "ipykernel_launcher":
        return False
    file_name = pathlib.Path(__file__).stem
    if arg0 == file_name:
        return True
    raise RuntimeError("cannot determine if running as a script or as a notebook")


if is_script():
    print("running as a script")
else:
    print("running as a notebook")
    from IPython import get_ipython
    from IPython.core.interactiveshell import InteractiveShell

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    # make a cell print all the outputs instead of just the last one
    InteractiveShell.ast_node_interactivity = "all"


from main import (  # noqa: E402
    DATASET_CATEGORY_CHOICES,
    DATASETSDIR,
    INPUT_IMAGE_RESOLUTION,
    STANDARD_CALLBACKS,
    evaluate,
    get_datamodule,
    get_global_seeder,
    get_model_dir,
    get_slurm_envvars,
    parser,
    test,
    train,
)

# In[]:

DEBUG = False
OFFLINE = False
print(f"{DEBUG=} {OFFLINE=}")


# In[]:

# Args
# The mechanism below allows the 'notebook mode' and the cli to
# eventually executed a subset of all dataset/category choices.

if is_script():
    cliargs = parser.parse_args()

else:
    # `dc` stands for "dataset-category"
    if DEBUG:
        argstr = "--dc " + " ".join(DATASET_CATEGORY_CHOICES[:1])
    else:
        argstr = "--dc " + " ".join(DATASET_CATEGORY_CHOICES)
    print(f"{argstr=}")
    cliargs = parser.parse_args(argstr.split())

print(f"{cliargs=}")

datasets_categories = cliargs.dataset_category
seed_global = cliargs.seed_global
seed_datamodule = cliargs.seed_datamodule

# defaults
global_seeder = get_global_seeder(seed_global)


# In[]:
# Model
# This function is different in each script.

from pytorch_lightning import Trainer  # noqa: E402

from anomalib.models import EfficientAd  # noqa: E402

if DEBUG:
    MODELNAME = "debug"
else:
    MODELNAME = "efficientad_wr101_s"


def get_model_trainer(logger=None):
    model = EfficientAd(
        teacher_out_channels=384,  # OK (default for small model)
        image_size=(INPUT_IMAGE_RESOLUTION, INPUT_IMAGE_RESOLUTION),
        model_size="small",
        lr=1e-4,  # OK (appendix A.1)
        weight_decay=1e-5,  # OK (appendix A.1), it has a decay at step 66.5k (95% of 70k)
        padding=False,  # deduced (see my notes)
        pad_maps=False,  # deduced (see my notes)
        batch_size=1,  # OK (appendix A.1)
        pretraining_images_dir=DATASETSDIR / "imagenet/data/train",
    )

    trainer = Trainer(
        enable_progress_bar=DEBUG and OFFLINE and not is_script(),
        devices=1,
        logger=logger,
        callbacks=STANDARD_CALLBACKS,
        accelerator="gpu",
        **(
            # DEBUG
            dict(
                max_steps=300,
            )
            if DEBUG
            else
            # PROD
            dict(
                max_steps=70000,
            )
        ),
    )
    return model, trainer


# just debug instanciating the model and trainer
if DEBUG:
    try:
        model, trainer = get_model_trainer()

    except Exception as ex:
        raise RuntimeError("failed to instanciate model and trainer") from ex

    else:
        del model, trainer
        import torch

        torch.cuda.empty_cache()


# In[]:
# Run

import torch  # noqa: E402
from progressbar import progressbar  # noqa: E402

from anomalib.data.utils import ValSplitMode  # noqa: E402
from anomalib.utils.loggers import AnomalibWandbLogger  # noqa: E402

MODELDIR = get_model_dir(MODELNAME)
print(f"{MODELDIR=}")

for ds_cat in progressbar(datasets_categories):
    dataset, category = ds_cat.split("/")
    print(f"{dataset=} {category=}")

    savedir = MODELDIR / dataset / category
    savedir.mkdir(exist_ok=True, parents=True)

    global_seeder()

    datamodule = get_datamodule(
        dataset,
        category,
        train_batch_size=(train_batch_size := 1),
        eval_batch_size=(eval_batch_size := 32),
        num_workers=(num_workers := 8),
        seed=seed_datamodule,
        # CUSTOM
        val_split_mode=ValSplitMode.FROM_TRAIN,
        val_split_ratio=0.5,
    )

    logger = AnomalibWandbLogger(
        save_dir=savedir,
        project="benchmark00",
        offline=OFFLINE,
    )
    logger.experiment.config.update(
        {
            "dataset": dataset,
            "category": category,
            "seed_global": seed_global,
            "seed_datamodule": seed_datamodule,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "num_workers": num_workers,
            "modelname": MODELNAME,
            **get_slurm_envvars(),
        }
    )

    try:
        model, trainer = get_model_trainer(logger=logger)
        logger.experiment.config.update({"model_class": model.__class__.__name__})
        train(dataset, category, datamodule, model, trainer, savedir)
        df_preds, asmaps, masks = test(dataset, category, datamodule, model, trainer, savedir)
        ascores = torch.as_tensor(df_preds["ascore"].values)
        imgclass = torch.as_tensor(df_preds["imgclass"].values)
        imgpaths = df_preds["imgpath"].values
        evaluate(imgpaths, ascores, imgclass, asmaps, masks, savedir, logger=logger, debug=DEBUG)

    except Exception as ex:
        print(str(ex))

    finally:
        logger.experiment.finish()

# %%
