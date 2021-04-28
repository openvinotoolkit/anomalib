from argparse import ArgumentParser

from pytorch_lightning import Trainer

from anomalib.datasets import MVTecDataModule
from anomalib.models.stfpm import Model

parser = ArgumentParser()
parser.add_argument("--dataroot", type=str, default="./datasets/MVTec")
parser.add_argument("--category", type=str, default="leather")

parser = Model.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

trainer = Trainer(max_epochs=100, gpus=1, check_val_every_n_epoch=50)
datamodule = MVTecDataModule(args.dataroot, args.category, args.batch_size, args.num_workers)
model = Model(args)
trainer.fit(model=model, datamodule=datamodule)
