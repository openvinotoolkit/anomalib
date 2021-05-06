from argparse import ArgumentParser
from pathlib import Path
from typing import List

import cv2
import torch
from pytorch_lightning import Trainer
from sklearn.metrics import roc_auc_score

from pytorch_lightning import Trainer
from anomalib.datasets import get_datamodule
from anomalib.models import get_model
from anomalib.models.stfpm import STFPMModel

from argparse import ArgumentParser

from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters
from anomalib.datasets import get_datamodule
from anomalib.models import get_model

# parser = ArgumentParser()
# parser.add_argument("--dataset", type=str, default="mvtec")
# parser.add_argument("--dataset_path", type=str, default="./datasets/MVTec/leather")
# parser.add_argument("--model", type=str, default="stfpm")
# parser.add_argument("--project_path", type=str, default="./results")
# parser.add_argument("--metric", type=str, default="auc")
# parser = STFPMModel.add_model_specific_args(parser)
# parser = Trainer.add_argparse_args(parser)
# args = parser.parse_args("")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")
    args = parser.parse_args()
    return args


args = get_args()
config = get_configurable_parameters(model_name=args.model, model_config_path=args.model_config_path)
datamodule = get_datamodule(config.dataset)
datamodule.setup()

model = get_model(config.model)
# TODO: load_from_checkpoint doesn't properly load the weights!!!
#  So instead, we use load_state_dict here. Investigate this.
# model.load_from_checkpoint(checkpoint_path="./results/weights/model.pth")
# model.load_state_dict(torch.load("./results/weights/model.ckpt")["state_dict"])
model.student_model.load_state_dict(torch.load("./results/weights/student_model.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device).eval()

model.student_model.eval()
model.teacher_model.eval()

trainer = Trainer(callbacks=model.callbacks, **config.trainer)
trainer.test(model=model, datamodule=datamodule)
# aucs: List[float] = []
#
# for i, batch in enumerate(datamodule.val_dataloader()):
#     output = model.test_step(batch, i)
#     aucs.append(output['auc'])
#     # images, mask = batch["image"], batch["mask"]
#     #
#     # teacher_features, student_features = model.forward(images.to(device))
#     # # teacher_features = model.teacher_model(images.to(device))
#     # # student_features = model.student_model(images.to(device))
#     # # loss = model.loss(teacher_features, student_features)
#     #
#     # anomaly_map = model.anomaly_map_generator(teacher_features, student_features)
#     # auc = roc_auc_score(mask.cpu().numpy().ravel(), anomaly_map.ravel())
#     # aucs.append(auc)
#     # image_path, mask_path = batch["image_path"][0], batch["mask_path"][0]
#     # images, masks = batch["image"], batch["mask"]
#     #
#     # defect_type = Path(image_path).parent.name
#     # image_filename = Path(image_path).stem
#     #
#     # original_image = cv2.imread(image_path)
#     # original_image = cv2.resize(original_image, (256, 256))
#     #
#     # heatmap_on_image = model.anomaly_map_generator.apply_heatmap_on_image(anomaly_map, original_image)
#     #
#     # cv2.imwrite(str(Path("./results/images/test") / f"{defect_type}_{image_filename}.jpg"), original_image)
#     # cv2.imwrite(str(Path("./results/images/test") / f"{defect_type}_{image_filename}_heatmap.jpg"), heatmap_on_image)
#     # cv2.imwrite(str(Path("./results/images/test") / f"{defect_type}_{image_filename}_mask.jpg"), masks.cpu().numpy())
#
# print(f"AUC: {torch.Tensor(aucs).mean()}")