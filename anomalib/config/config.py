"""Get configurable parameters."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# TODO: This would require a new design.
# TODO: https://jira.devtools.intel.com/browse/IAAALD-149

import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from warnings import warn

from omegaconf import DictConfig, ListConfig, OmegaConf

CONFIG_KEY_PROJECT_UNIQUEDIR = "unique_dir"
_CONFIGHASH_KEYS = [
    "dataset",
    "model",
    "project.seed",
    # trainer
    "trainer.accelerator",
    "trainer.accumulate_grad_batches",
    "trainer.auto_lr_find",
    "trainer.auto_scale_batch_size",
    "trainer.benchmark",
    "trainer.deterministic",
    "trainer.fast_dev_run",
    "trainer.gradient_clip_val",
    "trainer.limit_predict_batches",
    "trainer.limit_test_batches",
    "trainer.limit_train_batches",
    "trainer.limit_val_batches",
    "trainer.max_epochs",
    "trainer.max_steps",
    "trainer.max_time",
    "trainer.min_epochs",
    "trainer.min_steps",
    "trainer.move_metrics_to_cpu",
    "trainer.multiple_trainloader_mode",
    "trainer.num_nodes",
    "trainer.num_processes",
    "trainer.overfit_batches",
    "trainer.plugins",
    "trainer.precision",
    "trainer.reload_dataloaders_every_n_epochs",
    "trainer.replace_sampler_ddp",
    "trainer.sync_batchnorm",
    # trainer (excluded, keeping in comments for reference)
    # "trainer.detect_anomaly",
    # "trainer.devices",
    # "trainer.enable_checkpointing",
    # "trainer.enable_model_summary",
    # "trainer.enable_progress_bar",
    # "trainer.gpus",
    # "trainer.ipus",
    # "trainer.log_every_n_steps",
    # "trainer.num_sanity_val_steps",
    # "trainer.profiler",
    # "trainer.tpu_cores",
    # "trainer.track_grad_norm",
    # "trainer.val_check_interval",
]
_CONFIGHASH_DIGEST_NUM_BYTES = 8


def _get_now_str(timestamp: float) -> str:
    """Standard format for datetimes is defined here."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")


def _get_confighash(dictconfig: DictConfig) -> str:
    """Standard hash digest for config dict is defined here (not all keys are included)."""

    def get_masked_copy(dicconf: DictConfig, selected_keys: List[str]) -> DictConfig:
        """Deal with nested keys."""

        nested_keys = set(key for key in selected_keys if "." in key)
        non_nested_keys = set(selected_keys) - nested_keys

        copy = OmegaConf.masked_copy(dicconf, non_nested_keys)

        for key in nested_keys:
            first_key, rest = key.split(".", 1)
            copy[first_key] = get_masked_copy(dicconf[first_key], [rest])

        return copy

    dictconfig = get_masked_copy(dictconfig, _CONFIGHASH_KEYS)

    return hashlib.blake2b(
        str(dictconfig).encode("utf-8"),
        digest_size=_CONFIGHASH_DIGEST_NUM_BYTES,
    ).hexdigest()


def update_input_size_config(config: Union[DictConfig, ListConfig]) -> Union[DictConfig, ListConfig]:
    """Update config with image size as tuple, effective input size and tiling stride.

    Convert integer image size parameters into tuples, calculate the effective input size based on image size
    and crop size, and set tiling stride if undefined.

    Args:
        config (Union[DictConfig, ListConfig]): Configurable parameters object

    Returns:
        Union[DictConfig, ListConfig]: Configurable parameters with updated values
    """
    # handle image size
    if isinstance(config.dataset.image_size, int):
        config.dataset.image_size = (config.dataset.image_size,) * 2

    config.model.input_size = config.dataset.image_size

    if "tiling" in config.dataset.keys() and config.dataset.tiling.apply:
        if isinstance(config.dataset.tiling.tile_size, int):
            config.dataset.tiling.tile_size = (config.dataset.tiling.tile_size,) * 2
        if config.dataset.tiling.stride is None:
            config.dataset.tiling.stride = config.dataset.tiling.tile_size

    return config


def update_nncf_config(config: Union[DictConfig, ListConfig]) -> Union[DictConfig, ListConfig]:
    """Set the NNCF input size based on the value of the crop_size parameter in the configurable parameters object.

    Args:
        config (Union[DictConfig, ListConfig]): Configurable parameters of the current run.

    Returns:
        Union[DictConfig, ListConfig]: Updated configurable parameters in DictConfig object.
    """
    crop_size = config.dataset.image_size
    sample_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    if "optimization" in config.keys():
        if "nncf" in config.optimization.keys():
            if "input_info" not in config.optimization.nncf.keys():
                config.optimization.nncf["input_info"] = {"sample_size": None}
            config.optimization.nncf.input_info.sample_size = [1, 3, *sample_size]
            if config.optimization.nncf.apply:
                if "update_config" in config.optimization.nncf:
                    return OmegaConf.merge(config, config.optimization.nncf.update_config)
    return config


def update_multi_gpu_training_config(config: Union[DictConfig, ListConfig]) -> Union[DictConfig, ListConfig]:
    """Updates the config to change learning rate based on number of gpus assigned.

    Current behaviour is to ensure only ddp accelerator is used.

    Args:
        config (Union[DictConfig, ListConfig]): Configurable parameters for the current run

    Raises:
        ValueError: If unsupported accelerator is passed

    Returns:
        Union[DictConfig, ListConfig]: Updated config
    """
    # validate accelerator
    if config.trainer.accelerator is not None:
        if config.trainer.accelerator.lower() != "ddp":
            if config.trainer.accelerator.lower() in ("dp", "ddp_spawn", "ddp2"):
                warn(
                    f"Using accelerator {config.trainer.accelerator.lower()} is discouraged. "
                    f"Please use one of [null, ddp]. Setting accelerator to ddp"
                )
                config.trainer.accelerator = "ddp"
            else:
                raise ValueError(
                    f"Unsupported accelerator found: {config.trainer.accelerator}. Should be one of [null, ddp]"
                )
    # Increase learning rate
    # since pytorch averages the gradient over devices, the idea is to
    # increase the learning rate by the number of devices
    if "lr" in config.model:
        # Number of GPUs can either be passed as gpus: 2 or gpus: [0,1]
        n_gpus: Union[int, List] = 1
        if "trainer" in config and "gpus" in config.trainer:
            n_gpus = config.trainer.gpus
        lr_scaler = n_gpus if isinstance(n_gpus, int) else len(n_gpus)
        config.model.lr = config.model.lr * lr_scaler
    return config


def get_configurable_parameters(
    model_name: Optional[str] = None,
    config_path: Optional[Union[Path, str]] = None,
    weight_file: Optional[str] = None,
    config_filename: Optional[str] = "config",
    config_file_extension: Optional[str] = "yaml",
) -> Union[DictConfig, ListConfig]:
    """Get configurable parameters.

    Args:
        model_name: Optional[str]:  (Default value = None)
        config_path: Optional[Union[Path, str]]:  (Default value = None)
        weight_file: Path to the weight file
        config_filename: Optional[str]:  (Default value = "config")
        config_file_extension: Optional[str]:  (Default value = "yaml")

    Returns:
        Union[DictConfig, ListConfig]: Configurable parameters in DictConfig object.
    """
    if model_name is None and config_path is None:
        raise ValueError(
            "Both model_name and model config path cannot be None! "
            "Please provide a model name or path to a config file!"
        )

    if config_path is None:
        config_path = Path(f"anomalib/models/{model_name}/{config_filename}.{config_file_extension}")

    config = OmegaConf.load(config_path)

    # keep track of the original config file because it will be modified
    config_original: DictConfig = config.copy()

    # Dataset Configs
    if "format" not in config.dataset.keys():
        config.dataset.format = "mvtec"

    config = update_input_size_config(config)

    # IMPORTANT
    # the hash digest must be computed after the modifications above and before the modifications below
    # those from above modify values that are used in the hash digest computation, while those from below don't
    confighash: str = _get_confighash(config)  # 8-byte long

    # Project Configs
    project_path = Path(config.project.path) / config.model.name / config.dataset.name

    # add category subfolder if needed
    if config.dataset.format.lower() in ("btech", "mvtec"):
        project_path = project_path / config.dataset.category

    # set to False by default for backward compatibility
    config.project.setdefault(CONFIG_KEY_PROJECT_UNIQUEDIR, False)

    if config.project.unique_dir:
        project_path = project_path / f"run.{confighash}.{_get_now_str(time.time())}"

    else:
        project_path = project_path / "run"
        warn(
            f"{CONFIG_KEY_PROJECT_UNIQUEDIR} is set to False. "
            "This does not ensure that your results will be written in an empty directory and you may overwrite files."
        )

    (project_path / "weights").mkdir(parents=True, exist_ok=True)
    (project_path / "images").mkdir(parents=True, exist_ok=True)
    # write the original config for eventual debug (modified config at the end of the function)
    (project_path / "config_original.yaml").write_text(OmegaConf.to_yaml(config_original))
    (project_path / "confighash.txt").write_text(confighash)

    config.project.path = str(project_path)

    # loggers should write to results/model/dataset/category/ folder
    config.trainer.default_root_dir = str(project_path)

    if weight_file:
        config.trainer.resume_from_checkpoint = weight_file

    config = update_nncf_config(config)

    # thresholding
    if "metrics" in config.keys():
        # NOTE: Deprecate this after v0.4.0.
        if "adaptive" in config.metrics.threshold.keys():
            warn("adaptive will be deprecated in favor of method in config.metrics.threshold in v0.4.0.")
            config.metrics.threshold.method = "adaptive" if config.metrics.threshold.adaptive else "manual"
        if "image_default" in config.metrics.threshold.keys():
            warn("image_default will be deprecated in favor of manual_image in config.metrics.threshold in v0.4.0.")
            config.metrics.threshold.manual_image = (
                None if config.metrics.threshold.adaptive else config.metrics.threshold.image_default
            )
        if "pixel_default" in config.metrics.threshold.keys():
            warn("pixel_default will be deprecated in favor of manual_pixel in config.metrics.threshold in v0.4.0.")
            config.metrics.threshold.manual_pixel = (
                None if config.metrics.threshold.adaptive else config.metrics.threshold.pixel_default
            )

    (project_path / "config.yaml").write_text(OmegaConf.to_yaml(config))

    return config
