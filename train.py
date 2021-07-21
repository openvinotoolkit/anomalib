"""
Anomalib Traning Script.
    This script reads the name of the model or config file
    from command line, train/test the anomaly model to get
    quantitative and qualitative results.
"""
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything

from anomalib.config.config import get_configurable_parameters
from anomalib.datasets import get_datamodule
from anomalib.loggers import get_logger
from anomalib.models import get_model


def get_args() -> Namespace:
    """
    Get command line arguments.

    Returns:
        Namespace: List of arguements.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="patchcore", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")

    return parser.parse_args()


# def get_config():
#     parser = ArgumentParser(description="ANOMALYDETECTION")
#     parser.add_argument("--phase", choices=["train", "test"], default="train")
#     parser.add_argument(
#         "--dataset_path", default=r"/home/sakcay/Projects/data/MVTec"
#     )  # 'D:\Dataset\mvtec_anomaly_detection')#
#     parser.add_argument("--category", default="carpet")
#     parser.add_argument("--num_epochs", default=1)
#     parser.add_argument("--batch_size", default=32)
#     parser.add_argument("--load_size", default=256)  # 256
#     parser.add_argument("--input_size", default=224)
#     parser.add_argument("--coreset_sampling_ratio", default=0.001)
#     parser.add_argument(
#         "--project_root_path", default=r"/home/sakcay/Projects/ote/anomalib/results/patchcore/mvtec/leather"
#     )  # 'D:\Project_Train_Results\mvtec_anomaly_detection\210624\test') #
#     parser.add_argument("--save_src_code", default=True)
#     parser.add_argument("--save_anomaly_map", default=True)
#     parser.add_argument("--n_neighbors", type=int, default=9)
#     args = parser.parse_args()
#     return args


if __name__ == "__main__":
    args = get_args()
    config = get_configurable_parameters(model_name=args.model, model_config_path=args.model_config_path)

    # c = get_config()

    if config.project.seed != 0:
        seed_everything(config.project.seed)

    datamodule = get_datamodule(config)
    model = get_model(config)
    logger = get_logger(config)

    trainer = Trainer(**config.trainer, logger=logger)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
