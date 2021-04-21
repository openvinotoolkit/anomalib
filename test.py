from argparse import ArgumentParser

from pytorch_lightning import Trainer

from data import MVTecDataModule
from model import StudentTeacherFeaturePyramidMatching

parser = ArgumentParser()
parser.add_argument("--dataroot", type=str, default="./datasets/MVTec")
parser.add_argument("--category", type=str, default="leather")

parser = StudentTeacherFeaturePyramidMatching.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args("")

data_module = MVTecDataModule(root=args.dataroot, category=args.category, batch_size=1)
model = StudentTeacherFeaturePyramidMatching(args)
model.load_from_checkpoint(checkpoint_path="lightning_logs/version_12/checkpoints/epoch=8-step=71.ckpt")
print(model)
