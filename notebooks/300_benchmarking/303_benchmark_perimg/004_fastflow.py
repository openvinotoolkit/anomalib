#!/usr/bin/env python
# coding: utf-8

# FastFlow

# In[]:
# Setup (pre-args)

from __future__ import annotations

import os

import pandas as pd
from IPython.core.interactiveshell import InteractiveShell

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"  # DEBUG

os.environ["WANDB_NOTEBOOK_NAME"] = __file__

# make a cell print all the outputs instead of just the last one
InteractiveShell.ast_node_interactivity = "all"

# adjust the number of significant digits
pd.set_option("display.precision", 5)


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

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")


from main import (  # noqa: E402
    DATASET_CATEGORY_CHOICES,
    INPUT_IMAGE_RESOLUTION,
    STANDARD_CALLBACKS,
    evaluate,
    get_datamodule,
    get_global_seeder,
    get_model_dir,
    parser,
    test,
    train,
)

# In[]:

DEBUG = True
OFFLINE = True


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
train_batch_size = 4
eval_batch_size = 4
num_workers = 8

global_seeder = get_global_seeder(seed_global)


# In[]:
# Model
# This function is different in each script.

from pytorch_lightning import Trainer  # noqa: E402

from functools import partial, update_wrapper
from types import MethodType

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from torch.optim import Optimizer, Adam

from anomalib.data import TaskType
from anomalib.models import Fastflow

from anomalib.models import Patchcore  # noqa: E402

if DEBUG:
    MODELNAME = "debug"
else:
    MODELNAME = "fastflow"


def get_model_trainer(logger=None):
# based on `anomalib/notebooks/200_models/201_fastflow.ipynb`
# https://github.com/openvinotoolkit/anomalib/blob/52bbb0b2417461ff097a819a06cc7ac3ff408149/notebooks/200_models/201_fastflow.ipynb

%autoreload 2

if model_name == "fastflow":


    if model_version != "default":
        raise ValueError(f"Unknown model version: `{model_version}` for `{model_name}`") 
    
    def configure_optimizers(lightning_module: LightningModule, optimizer: Optimizer):
        """Override to customize the LightningModule.configure_optimizers` method."""
        return optimizer

    model = Fastflow(
        input_size=(INPUT_IMAGE_RESOLUTION, INPUT_IMAGE_RESOLUTION), 
        # defaults from
        # https://github.com/openvinotoolkit/anomalib/blob/52bbb0b2417461ff097a819a06cc7ac3ff408149/src/anomalib/models/fastflow/config.yaml
        backbone="resnet18", 
        pre_trained=True,
        flow_steps=8,
        hidden_ratio=1.0,
        conv3x3_only=True,
    )

    fn = partial(
        configure_optimizers, 
        optimizer=Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
    )
    update_wrapper(fn, configure_optimizers)  # necessary for `is_overridden`
    model.configure_optimizers = MethodType(fn, model)

    trainer = Trainer(
        logger=logger,
        callbacks=[
            METRICS_CALLBACK, POSTPROCESSING_CALLBACK,
            ModelCheckpoint(mode="max", monitor="pixel_AUROC",),
            EarlyStopping(monitor="pixel_AUROC", mode="max", patience=3,),
        ],
        accelerator="auto",  
        devices=1,
        max_epochs=1,  # DEBUG  
        # max_epochs=100,  # PROD
    )
    return model, trainer


# just debug instanciating the model and trainer
try:
    model, trainer = get_model_trainer()
    print(f"model={model.__class__.__name__}")

except Exception as ex:
    raise RuntimeError("failed to instanciate model and trainer") from ex


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
            "modelname": MODELNAME,
        }
    )

    try:
        model, trainer = get_model_trainer(logger=logger)
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
