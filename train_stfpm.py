import argparse
import glob
import os
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18

from anomalib.datasets import MVTecDataModule
from anomalib.models.stfpm import FeatureExtractor, FeaturePyramidLoss, AnomalyMapGenerator
from torchvision import transforms

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


# def data_transforms_inv():
#     data_transforms_inv = transforms.Compose([transforms.Normalize(mean=list(-np.divide(mean_train, std_train)), std=list(np.divide(1, std_train)))])
#     return data_transforms_inv


# def cal_loss(fs_list, ft_list, criterion):
#     tot_loss = 0
#     for i in range(len(ft_list)):
#         fs = fs_list[i]
#         ft = ft_list[i]
#         _, _, h, w = fs.shape
#         fs_norm = torch.div(fs, torch.norm(fs, p=2, dim=1, keepdim=True))
#         ft_norm = torch.div(ft, torch.norm(ft, p=2, dim=1, keepdim=True))
#         f_loss = (0.5 / (w * h)) * criterion(fs_norm, ft_norm)
#         tot_loss += f_loss
#     return tot_loss


# def cal_anomaly_map(fs_list, ft_list, out_size=256):
#     pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
#     anomaly_map = np.ones([out_size, out_size])
#     a_map_list = []
#     for i in range(len(ft_list)):
#         fs = fs_list[i]
#         ft = ft_list[i]
#         fs_norm = torch.div(fs, torch.norm(fs, p=2, dim=1, keepdim=True))
#         ft_norm = torch.div(ft, torch.norm(ft, p=2, dim=1, keepdim=True))
#         a_map = 0.5 * pdist(fs_norm, ft_norm) ** 2
#         a_map = F.interpolate(a_map, size=out_size, mode="bilinear")
#         a_map = a_map[0, 0, :, :].to("cpu").detach().numpy()  # check
#         a_map_list.append(a_map)
#         anomaly_map *= a_map
#     return anomaly_map, a_map_list
#
#
# def show_cam_on_image(img, anomaly_map):
#     heatmap = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
#     cam = np.float32(heatmap) + np.float32(img)
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam)
#
#
# def cvt2heatmap(gray):
#     heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
#     return heatmap
#
#
# def heatmap_on_image(heatmap, image):
#     out = np.float32(heatmap) / 255 + np.float32(image) / 255
#     out = out / np.max(out)
#     return np.uint8(255 * out)
#
#
# def min_max_norm(image):
#     a_min, a_max = image.min(), image.max()
#     return (image - a_min) / (a_max - a_min)


class STFPM:
    def __init__(self, hparams: Namespace):
        self.hparams = hparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_transform = data_transforms(input_size=hparams.input_size, mean_train=mean_train, std_train=std_train)
        self.data_module = self.load_dataset()
        # self.load_model()

        self.student_model = FeatureExtractor(resnet18(pretrained=False), ["layer1", "layer2", "layer3"])
        self.teacher_model = FeatureExtractor(resnet18(pretrained=True), ["layer1", "layer2", "layer3"])
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False

        self.loss = FeaturePyramidLoss()
        self.anomaly_generator = AnomalyMapGenerator(image_size=hparams.input_size)

    def load_dataset(self) -> MVTecDataModule:
        # image_datasets = datasets.ImageFolder(
        #     root=self.hparams.dataset_path / self.hparams.category, transform=self.data_transform
        # )
        # self.dataloaders = DataLoader(image_datasets, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)
        # dataset_sizes = {"train": len(image_datasets)}
        # print("Dataset size : Train set - {}".format(dataset_sizes["train"]))
        data_module = MVTecDataModule(
            root=self.hparams.dataset_path,
            category=self.hparams.category,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        data_module.setup()
        print(f"Loading MVTec {self.hparams.category} category")
        return data_module

    # def load_model(self):
    #     self.features_t = []
    #     self.features_s = []
    #
    #     def hook_t(module, input, output):
    #         self.features_t.append(output)
    #
    #     def hook_s(module, input, output):
    #         self.features_s.append(output)
    #
    #     self.model_t = resnet18(pretrained=True).to(self.device)
    #     self.model_t.layer1[-1].register_forward_hook(hook_t)
    #     self.model_t.layer2[-1].register_forward_hook(hook_t)
    #     self.model_t.layer3[-1].register_forward_hook(hook_t)
    #
    #     self.model_s = resnet18(pretrained=False).to(self.device)
    #     self.model_s.layer1[-1].register_forward_hook(hook_s)
    #     self.model_s.layer2[-1].register_forward_hook(hook_s)
    #     self.model_s.layer3[-1].register_forward_hook(hook_s)

    def train(self):

        # self.criterion = torch.nn.MSELoss(reduction="sum")
        # optimizer = torch.optim.SGD(
        #     self.model_s.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=0.0001
        # )
        optimizer = torch.optim.SGD(
            self.student_model.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=0.0001
        )

        # self.load_dataset()

        self.teacher_model.to(self.device).eval()
        self.student_model.to(self.device).train()

        # start_time = time.time()
        # global_step = 0

        for epoch in range(self.hparams.num_epochs):
            # print("-" * 20)
            # print("Epoch {}/{}".format(epoch, self.hparams.num_epochs - 1))
            # print("-" * 20)

            # self.model_t.eval()
            # self.model_s.train()

            # for idx, (batch, _) in enumerate(self.dataloaders):  # batch loop
            for idx, batch in enumerate(self.data_module.train_dataloader()):
                # global_step += 1
                images = batch["image"].to(self.device)
                optimizer.zero_grad()
                # with torch.set_grad_enabled(True):
                #     self.features_t = []
                #     self.features_s = []
                #     _ = self.model_t(images)
                #     _ = self.model_s(images)

                teacher_features = self.teacher_model(images)
                student_features = self.student_model(images)

                # get loss using features.
                # loss = cal_loss(self.features_s, self.features_t, self.criterion)
                loss = self.loss(teacher_features, student_features)
                loss.backward()
                optimizer.step()

                if idx % 2 == 0:
                    print(f"Epoch : {epoch} | Loss : {loss.data:.4f}")

        # print("Total time consumed : {}".format(time.time() - start_time))
        print(">> Saving the weights...")
        # torch.save(self.model_s.state_dict(), self.hparams.weight_path / "student_model.pth")
        torch.save(self.student_model.state_dict(), self.hparams.weight_path / "student_model.pth")

    def test(self):
        print(">> Testing the model")
        try:
            # self.model_s.load_state_dict(torch.load(self.hparams.weight_path / "student_model.pth"))
            self.student_model.load_state_dict(torch.load(self.hparams.weight_path / "student_model.pth"))
        except ValueError as error:
            print(error)
            print("Model weight file not found. Cannot load the model.")
        # self.model_t.eval()
        # self.model_s.eval()

        self.student_model.to(self.device).eval()
        self.teacher_model.to(self.device).eval()

        # # test_path = os.path.join(self.hparams.dataset_path, "test")
        # test_path = str(self.hparams.dataset_path / self.hparams.category / "test")
        # gt_path = str(self.hparams.dataset_path / self.hparams.category / "ground_truth")
        # # gt_path = os.path.join(self.hparams.dataset_path, "ground_truth")
        # test_imgs = glob.glob(test_path + "/**/*.png", recursive=True)
        # test_imgs = [i for i in test_imgs if "good" not in i]
        # gt_imgs = glob.glob(gt_path + "/**/*.png", recursive=True)
        # test_imgs.sort()
        # gt_imgs.sort()
        true_mask_list = []
        pred_mask_list = []

        # for i in range(len(test_imgs)):
        for i, batch in enumerate(self.data_module.val_dataloader()):
            # test_img_path = test_imgs[i]
            # gt_img_path = gt_imgs[i]
            image_path, mask_path = batch['image_path'][0], batch['mask_path'][0]
            images, masks = batch['image'], batch['mask']
            # assert (
            #     os.path.split(image_path)[1].split(".")[0] == os.path.split(mask_path)[1].split("_")[0]
            # ), "Something wrong with test and ground truth pair!"
            defect_type = os.path.split(os.path.split(image_path)[0])[1]
            image_filename = os.path.split(image_path)[1].split(".")[0]

            # # ground truth
            gt_img_o = cv2.imread(mask_path, 0)
            gt_img_o = cv2.resize(gt_img_o, (self.hparams.input_size, self.hparams.input_size))
            # # gt_val_list.extend(gt_img_o.ravel() // 255)
            #
            # # load image
            test_img_o = cv2.imread(image_path)
            test_img_o = cv2.resize(test_img_o, (self.hparams.input_size, self.hparams.input_size))
            # test_img = Image.fromarray(test_img_o)
            # test_img = self.data_transform(test_img)
            # test_img = torch.unsqueeze(test_img, 0).to(self.device)
            # with torch.set_grad_enabled(False):
            #     self.features_t = []
            #     self.features_s = []
            #     _ = self.model_t(test_img)
            #     _ = self.model_s(test_img)
            # get anomaly map & each features

            # self.features_t = self.teacher_model(test_img)
            # self.features_s = self.student_model(test_img)
            # images = batch['image'].to(self.device)
            teacher_features = self.teacher_model(images.to(self.device))
            student_features = self.student_model(images.to(self.device))

            # self.features_t = [f for f in self.features_t.values()]
            # self.features_s = [f for f in self.features_s.values()]
            # anomaly_map, a_maps = cal_anomaly_map(self.features_s, self.features_t, out_size=self.hparams.input_size)

            anomaly_map = self.anomaly_generator.compute_anomaly_map(teacher_features, student_features)
            heatmap = self.anomaly_generator.compute_heatmap(anomaly_map)
            hm_on_img = self.anomaly_generator.apply_heatmap_on_image(heatmap, test_img_o)

            true_mask_list.extend(masks.numpy().ravel())
            pred_mask_list.extend(anomaly_map.ravel())

            # # normalize anomaly amp
            # anomaly_map_norm = min_max_norm(anomaly_map)
            # # anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm * 255)
            # # # 64x64 map
            # # am64 = min_max_norm(a_maps[0])
            # # am64 = cvt2heatmap(am64 * 255)
            # # # 32x32 map
            # # am32 = min_max_norm(a_maps[1])
            # # am32 = cvt2heatmap(am32 * 255)
            # # # 16x16 map
            # # am16 = min_max_norm(a_maps[2])
            # # am16 = cvt2heatmap(am16 * 255)
            # # anomaly map on image
            # heatmap = cvt2heatmap(anomaly_map_norm * 255)
            # hm_on_img = heatmap_on_image(heatmap, test_img_o)

            # save images
            cv2.imwrite(str(self.hparams.sample_path / f"{defect_type}_{image_filename}.jpg"), test_img_o)
            # cv2.imwrite(os.path.join(self.hparams.sample_path, f"{defect_type}_{image_filename}.jpg"), test_img_o)
            # cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_am64.jpg'), am64)
            # cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_am32.jpg'), am32)
            # cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_am16.jpg'), am16)
            # cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_amap.jpg'), anomaly_map_norm_hm)
            cv2.imwrite(str(self.hparams.sample_path / f"{defect_type}_{image_filename}_heatmap.jpg"), hm_on_img)
            # cv2.imwrite(os.path.join(self.hparams.sample_path, f"{defect_type}_{image_filename}_amap_on_img.jpg"), hm_on_img)
            cv2.imwrite(str(self.hparams.sample_path / f"{defect_type}_{image_filename}_mask.jpg"), gt_img_o)
            # cv2.imwrite(os.path.join(self.hparams.sample_path, f"{defect_type}_{image_filename}_gt.jpg"), gt_img_o)

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
