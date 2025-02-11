# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Example showing how to use advanced logging features in Anomalib.

This example demonstrates how to configure different loggers (TensorBoard,
WandB, MLflow, Comet) and customize logging behavior.
"""

from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.loggers import AnomalibMLFlowLogger, AnomalibTensorBoardLogger, AnomalibWandbLogger
from anomalib.models import Patchcore

# 1. Basic TensorBoard Logging
# This is the default logger
engine = Engine(
    logger=AnomalibTensorBoardLogger(save_dir="logs/tensorboard"),
    max_epochs=1,
)

# 2. Weights & Biases (WandB) Logging
# Track experiments with WandB
engine = Engine(
    logger=AnomalibWandbLogger(
        project="anomalib",
        name="patchcore_experiment",
        save_dir="logs/wandb",
    ),
    max_epochs=1,
)

# 3. MLflow Logging
# Track experiments with MLflow
engine = Engine(
    logger=AnomalibMLFlowLogger(
        experiment_name="anomalib",
        tracking_uri="logs/mlflow",
    ),
    max_epochs=1,
)

# 4. Multiple Loggers
# Use multiple loggers simultaneously
engine = Engine(
    logger=[
        AnomalibTensorBoardLogger(save_dir="logs/tensorboard"),
        AnomalibWandbLogger(project="anomalib", save_dir="logs/wandb"),
    ],
    max_epochs=1,
)

# 5. Complete Training Example with Logging
model = Patchcore()
datamodule = MVTecAD(
    root=Path("./datasets/MVTecAD"),
    category="bottle",
)

# Configure engine with logging
engine = Engine(
    logger=AnomalibTensorBoardLogger(save_dir="logs/tensorboard"),
    max_epochs=1,
    log_graph=True,  # Log model graph
    enable_checkpointing=True,  # Save model checkpoints
    default_root_dir="results",  # Root directory for all outputs
)

# Train with logging enabled
engine.fit(
    model=model,
    datamodule=datamodule,
)
