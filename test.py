from argparse import ArgumentParser
from pathlib import Path
from typing import List

import cv2
import torch
from pytorch_lightning import Trainer
from sklearn.metrics import roc_auc_score

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
args = parser.parse_args("")

datamodule = get_datamodule(args)
datamodule.setup()

model = get_model(args)
# TODO: load_from_checkpoint doesn't properly load the weights!!!
#  So instead, we use load_state_dict here. Investigate this.
# model.load_from_checkpoint(checkpoint_path="./results/weights/model.pth")
model.load_state_dict(torch.load("./results/weights/model.ckpt")["state_dict"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

aucs: List[float] = []

for i, batch in enumerate(datamodule.val_dataloader()):
    image_path, mask_path = batch["image_path"][0], batch["mask_path"][0]
    images, masks = batch["image"], batch["mask"]

    defect_type = Path(image_path).parent.name
    image_filename = Path(image_path).stem

    # load image
    # TODO: Use the image tensor instead, without re-reading it here!
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (256, 256))

    teacher_features, student_features = model(images.to(device))

    anomaly_map = model.anomaly_map_generator.compute_anomaly_map(teacher_features, student_features)
    heatmap_on_image = model.anomaly_map_generator.apply_heatmap_on_image(anomaly_map, original_image)

    aucs.append(roc_auc_score(masks.numpy().ravel(), anomaly_map.ravel()))
#
#     # save images
#     cv2.imwrite(str(Path("./results") / f"{defect_type}_{image_filename}.jpg"), original_image)
#     cv2.imwrite(str(Path("./results") / f"{defect_type}_{image_filename}_heatmap.jpg"), heatmap_on_image)
#     cv2.imwrite(str(Path("./results") / f"{defect_type}_{image_filename}_mask.jpg"), masks.numpy())

print(f"AUC: {torch.Tensor(aucs).mean()}")
