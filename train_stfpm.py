import argparse
import os
from argparse import Namespace
from pathlib import Path

import cv2
import torch
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torchvision.models import resnet18

from anomalib.datasets import MVTecDataModule
from anomalib.models.stfpm import FeatureExtractor, FeaturePyramidLoss, AnomalyMapGenerator

# imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]


def data_transforms(input_size=256, mean_train=mean_train, std_train=std_train):
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Normalize(mean=mean_train, std=std_train),
        ]
    )
    return data_transforms


class STFPM:
    def __init__(self, hparams: Namespace):
        self.hparams = hparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_transform = data_transforms(input_size=hparams.input_size, mean_train=mean_train, std_train=std_train)
        self.data_module = self.load_dataset()

        self.student_model = FeatureExtractor(resnet18(pretrained=False), ["layer1", "layer2", "layer3"])
        self.teacher_model = FeatureExtractor(resnet18(pretrained=True), ["layer1", "layer2", "layer3"])
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False

        self.loss = FeaturePyramidLoss()
        self.anomaly_generator = AnomalyMapGenerator(image_size=hparams.input_size)

    def load_dataset(self) -> MVTecDataModule:
        data_module = MVTecDataModule(
            root=self.hparams.dataset_path,
            category=self.hparams.category,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        data_module.setup()
        print(f"Loading MVTec {self.hparams.category} category")
        return data_module

    def train(self):
        optimizer = torch.optim.SGD(
            self.student_model.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=0.0001
        )

        self.teacher_model.to(self.device).eval()
        self.student_model.to(self.device).train()

        for epoch in range(self.hparams.num_epochs):
            for idx, batch in enumerate(self.data_module.train_dataloader()):
                images = batch["image"].to(self.device)
                optimizer.zero_grad()

                teacher_features = self.teacher_model(images)
                student_features = self.student_model(images)

                loss = self.loss(teacher_features, student_features)
                loss.backward()
                optimizer.step()

                if idx % 2 == 0:
                    print(f"Epoch : {epoch} | Loss : {loss.data:.4f}")

        print(">> Saving the weights...")
        torch.save(self.student_model.state_dict(), self.hparams.weight_path / "student_model.pth")

    def test(self):
        print(">> Testing the model")
        try:
            self.student_model.load_state_dict(torch.load(self.hparams.weight_path / "student_model.pth"))
        except ValueError as error:
            print(error)
            print("Model weight file not found. Cannot load the model.")

        self.student_model.to(self.device).eval()
        self.teacher_model.to(self.device).eval()

        true_mask_list = []
        pred_mask_list = []

        for i, batch in enumerate(self.data_module.val_dataloader()):

            image_path, mask_path = batch["image_path"][0], batch["mask_path"][0]
            images, masks = batch["image"], batch["mask"]

            defect_type = os.path.split(os.path.split(image_path)[0])[1]
            image_filename = os.path.split(image_path)[1].split(".")[0]

            # load image
            original_image = cv2.imread(image_path)
            original_image = cv2.resize(original_image, (self.hparams.input_size, self.hparams.input_size))

            teacher_features = self.teacher_model(images.to(self.device))
            student_features = self.student_model(images.to(self.device))

            anomaly_map = self.anomaly_generator.compute_anomaly_map(teacher_features, student_features)
            heatmap = self.anomaly_generator.compute_heatmap(anomaly_map)
            heatmap_on_image = self.anomaly_generator.apply_heatmap_on_image(heatmap, original_image)

            true_mask_list.extend(masks.numpy().ravel())
            pred_mask_list.extend(anomaly_map.ravel())

            # save images
            cv2.imwrite(str(self.hparams.sample_path / f"{defect_type}_{image_filename}.jpg"), original_image)
            cv2.imwrite(str(self.hparams.sample_path / f"{defect_type}_{image_filename}_heatmap.jpg"), heatmap_on_image)
            cv2.imwrite(str(self.hparams.sample_path / f"{defect_type}_{image_filename}_mask.jpg"), masks.numpy())

        print(f"AUC: {roc_auc_score(true_mask_list, pred_mask_list)}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="train")
    parser.add_argument("--project_path", type=Path, default="./results/")
    parser.add_argument("--dataset_path", type=Path, default="./datasets/MVTec")
    parser.add_argument("--category", default="zipper")
    parser.add_argument("--num_epochs", default=10)
    parser.add_argument("--lr", default=0.4)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--num_workers", default=36)
    parser.add_argument("--input_size", default=256)
    args = parser.parse_args()

    args.weight_path = args.project_path / "weights"
    args.sample_path = args.project_path / "images"

    args.project_path.mkdir(exist_ok=True)
    args.weight_path.mkdir(exist_ok=True)
    args.sample_path.mkdir(exist_ok=True)
    return args


if __name__ == "__main__":

    args = get_args()
    # phase = args.phase
    # dataset_path = args.dataset_path
    # category = dataset_path.split('\\')[-1]
    # num_epochs = args.num_epoch
    # lr = args.lr
    # batch_size = args.batch_size
    # save_weight = args.save_weight
    # input_size = args.input_size
    # project_path = args.project_path
    # sample_path = os.path.join(project_path, 'sample')
    # Create dirs.

    # os.makedirs(sample_path, exist_ok=True)
    # weight_save_path = os.path.join(project_path, "saved")
    # os.makedirs(weight_save_path, exist_ok=True)

    model = STFPM(hparams=args)
    if args.phase == "train":
        model.train()
        model.test()
    elif args.phase == "test":
        model.test()
    else:
        print("Phase argument must be train or test.")
