from argparse import ArgumentParser

from pytorch_lightning import Trainer

from anomalib.datasets import get_datamodule
from anomalib.models import get_model
from anomalib.models.stfpm import STFPMModel

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="mvtec")
parser.add_argument("--dataset_path", type=str, default="./datasets/MVTec/zipper")
parser.add_argument("--model", type=str, default="stfpm")
parser.add_argument("--project_path", type=str, default="./results")
parser.add_argument("--metric", type=str, default="auc")

parser = STFPMModel.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()


datamodule = get_datamodule(args)
model = get_model(args)
trainer = Trainer(max_epochs=args.num_epochs, gpus=1, check_val_every_n_epoch=2, callbacks=model.callbacks)

trainer.fit(model=model, datamodule=datamodule)
