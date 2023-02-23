from __future__ import annotations

import os
from pathlib import Path

from git.repo import Repo

current_directory = Path.cwd()
if current_directory.parent.name == "500_use_cases" and current_directory.name == "501_Dobot":
    # On the assumption that, the notebook is located in
    #   ~/anomalib/notebooks/500_use_cases/dobot
    root_directory = current_directory.parent.parent.parent
elif current_directory.name == "anomalib":
    # This means that the notebook is run from the main anomalib directory.
    root_directory = current_directory
else:
    # Otherwise, we'll need to clone the anomalib repo to the `current_directory`
    repo = Repo.clone_from(url="https://github.com/openvinotoolkit/anomalib.git", to_path=current_directory)
    root_directory = current_directory / "anomalib"

os.chdir(root_directory)
notebook_path = root_directory / "notebooks" / "500_use_cases" / "501_Dobot"
dataset_path = root_directory / "datasets" / "cubes"

from anomalib.config import get_configurable_parameters
from pytorch_lightning import Trainer
from matplotlib import pyplot as plt
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
MODEL = "padim"  # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'
CONFIG_PATH = notebook_path /"cubes_config.yaml"
with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as file:
    print(file.read())
config = get_configurable_parameters(config_path=CONFIG_PATH)
config["dataset"]["path"] = str(dataset_path) # or wherever the Custom Dataset is stored.
print(config)
from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType


datamodule = Folder(
    root=dataset_path,
    normal_dir="normal",
    abnormal_dir="abnormal",
    normal_split_ratio=0.2,
    image_size=(256, 256),
    train_batch_size=32,
    eval_batch_size=32,
    task=TaskType.CLASSIFICATION,
)
datamodule.setup()          # Split the data to train/val/test/prediction sets.
datamodule.prepare_data()   # Create train/val/test/predic dataloaders

i, data = next(enumerate(datamodule.val_dataloader()))
print(data.keys())
from anomalib.models import Padim


model = Padim(
    input_size=(256, 256),
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
)
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.callbacks.export import ExportCallback, ExportMode
from pytorch_lightning.callbacks import ModelCheckpoint


callbacks = [
    MetricsConfigurationCallback(
        task=TaskType.CLASSIFICATION,
        image_metrics=["AUROC"],
    ),
    ModelCheckpoint(
        mode="max",
        monitor="image_AUROC",
    ),
    PostProcessingConfigurationCallback(
        normalization_method=NormalizationMethod.MIN_MAX,
        threshold_method=ThresholdMethod.ADAPTIVE,
    ),
    ExportCallback(
        input_size=(256, 256),
        dirpath=str(notebook_path),
        filename="model",
        export_mode=ExportMode.OPENVINO,
    ),
]
trainer = Trainer(
  callbacks=callbacks,
  accelerator= "auto",
  auto_scale_batch_size= False,
  check_val_every_n_epoch= 1,
  devices= 1,
  gpus= None,
  max_epochs= 1,
  num_sanity_val_steps= 0,
  val_check_interval= 1.0,
)
trainer.fit(model=model, datamodule=datamodule)
# Validation
test_results = trainer.test(model=model, datamodule=datamodule)