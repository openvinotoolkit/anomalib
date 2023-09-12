#!/usr/bin/env python
# coding: utf-8

# FastFlow WideResnet50

# In[]:
# Setup (pre-args)

from __future__ import annotations

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
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

DEBUG = True
DEBUG_PARAMS = False
OFFLINE = False
print(f"{DEBUG=} {DEBUG_PARAMS=} {OFFLINE=}")


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
# 32 in the paper but it would break the memory, 
# so we use 16 with 2x gradient accumulation
train_batch_size = 16
eval_batch_size = 16
num_workers = 8

global_seeder = get_global_seeder(seed_global)


# In[]:
# Model
# This function is different in each script.

from functools import partial, update_wrapper  # noqa: E402
from types import MethodType  # noqa: E402

from pytorch_lightning import LightningModule, Trainer  # noqa: E402
from pytorch_lightning.callbacks import GradientAccumulationScheduler  # noqa: E402
from torch.optim import Adam, Optimizer  # noqa: E402

from anomalib.models import Fastflow  # noqa: E402

if DEBUG:
    MODELNAME = "debug"
else:
    MODELNAME = "fastflow_wr50"


def get_model_trainer(logger=None):
    # based on `anomalib/notebooks/200_models/201_fastflow.ipynb`
    # https://github.com/openvinotoolkit/anomalib/blob/52bbb0b2417461ff097a819a06cc7ac3ff408149/notebooks/200_models/201_fastflow.ipynb

    def configure_optimizers(lightning_module: LightningModule, optimizer: Optimizer):
        """Override to customize the LightningModule.configure_optimizers` method."""
        return optimizer

    model_params = (
        # DEBUG
        dict(
            backbone="wide_resnet50_2",
            flow_steps=1,
            conv3x3_only=False,
            hidden_ratio=0.1,
        )
        if DEBUG and DEBUG_PARAMS
        else
        # PROD
        dict(
            backbone="wide_resnet50_2",
            flow_steps=8,
            conv3x3_only=False,  # OK
            hidden_ratio=1.0,  # OK
        )
    )

    model = Fastflow(
        input_size=(INPUT_IMAGE_RESOLUTION, INPUT_IMAGE_RESOLUTION),
        pre_trained=True,
        **model_params,
    )

    fn = partial(
        configure_optimizers, optimizer=Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
    )
    update_wrapper(fn, configure_optimizers)  # necessary for `is_overridden`
    model.configure_optimizers = MethodType(fn, model)

    trainer_params = (
        # DEBUG
        dict(
            accelerator="gpu",
            devices=1,
            max_epochs=5,
        )
        if DEBUG else
        # PROD
        dict(
            accelerator="gpu",
            devices=1,
            max_epochs=500,  # OK
        )
    )

    trainer = Trainer(
        enable_progress_bar=DEBUG and OFFLINE and not is_script(),
        logger=logger,
        callbacks=STANDARD_CALLBACKS + [
            GradientAccumulationScheduler(scheduling={0: 2})
        ],
        **trainer_params,
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
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed_datamodule,
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
        if DEBUG: from time import time; ts = time(); print(f"train start {ts=}")
        train(dataset, category, datamodule, model, trainer, savedir)
        if DEBUG: te = time(); sec = te - ts; print(f"train end {te=} {sec=}")
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