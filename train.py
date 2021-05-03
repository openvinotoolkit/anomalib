from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from anomalib.datasets import get_datamodule
from anomalib.models import get_model

# parser = ArgumentParser()
# parser.add_argument("--dataset", type=str, default="mvtec")
# parser.add_argument("--dataset_path", type=str, default="./datasets/MVTec/zipper")
# parser.add_argument("--model", type=str, default="stfpm")
# parser.add_argument("--project_path", type=str, default="./results")
# parser.add_argument("--metric", type=str, default="auc")
#
# parser = STFPMModel.add_model_specific_args(parser)
# parser = Trainer.add_argparse_args(parser)
# args = parser.parse_args()


hparams = OmegaConf.load("./params.yaml")

datamodule = get_datamodule(hparams)
model = get_model(hparams)
trainer = Trainer(
    logger=hparams.logger,
    # checkpoint_callback,
    callbacks=model.callbacks,
    # default_root_dir,
    # gradient_clip_val,
    # process_position,
    # num_nodes,
    # num_processes,
    gpus=hparams.gpus,
    # auto_select_gpus,
    # tpu_cores,
    # log_gpu_memory,
    # progress_bar_refresh_rate,
    # overfit_batches,
    # track_grad_norm,
    # check_val_every_n_epoch,
    # fast_dev_run,
    # accumulate_grad_batches,
    max_epochs=hparams.max_epochs,
    # min_epochs,
    # max_steps,
    # min_steps,
    # limit_train_batches,
    # limit_val_batches,
    # limit_test_batches,
    # limit_predict_batches,
    # val_check_interval,
    # flush_logs_every_n_steps,
    # log_every_n_steps,
    # accelerator,
    # sync_batchnorm,
    # precision,
    # weights_summary,
    # weights_save_path,
    # num_sanity_val_steps,
    # truncated_bptt_steps,
    # resume_from_checkpoint,
    # profiler: Optional[Union[BaseProfiler, bool, str]] = None,
    # benchmark: bool = False,
    # deterministic: bool = False,
    # reload_dataloaders_every_epoch: bool = False,
    # auto_lr_find: Union[bool, str] = False,
    # replace_sampler_ddp: bool = True,
    # terminate_on_nan: bool = False,
    # auto_scale_batch_size: Union[str, bool] = False,
    # prepare_data_per_node: bool = True,
    # plugins: Optional[Union[Plugin, str, list]] = None,
    # amp_backend: str = 'native',
    # amp_level: str = 'O2',
    # distributed_backend: Optional[str] = None,
    # automatic_optimization: Optional[bool] = None,
    # move_metrics_to_cpu: bool = False,
    # enable_pl_optimizer: bool = None,  # todo: remove in v1.3
    # multiple_trainloader_mode: str = 'max_size_cycle',
    # stochastic_weight_avg: bool = False,
)

trainer.fit(model=model, datamodule=datamodule)
