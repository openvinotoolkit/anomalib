import argparse
import torchvision
import pytorch_lightning as pl
import os
import pickle
import random
from random import sample
from torch import Tensor
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from collections import OrderedDict

# import tarfile
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from skimage import morphology
from skimage.segmentation import mark_boundaries
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models import wide_resnet50_2, resnet18
from tqdm import tqdm

import datasets.mvtec as mvtec
from anomalib.datasets.mvtec import MVTecDataModule
from anomalib.models.shared.feature_extractor import FeatureExtractor

# import urllib.request

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# set transforms
transform_x = T.Compose(
    [
        T.Resize(256, Image.ANTIALIAS),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
transform_mask = T.Compose([T.Resize(256, Image.NEAREST), T.CenterCrop(224), T.ToTensor()])


def parse_args():
    parser = argparse.ArgumentParser("PaDiM")
    parser.add_argument("--data_path", type=str, default="/home/sakcay/Projects/ote/anomalib/datasets/MVTec")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--arch", type=str, choices=["resnet18", "wide_resnet50_2"], default="resnet18")
    return parser.parse_args()


class MultiVariateGaussian(torch.nn.Module):
    @staticmethod
    def _cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
        """Estimates covariance matrix like numpy.cov"""
        # ensure at least 2D
        if x.dim() == 1:
            x = x.view(-1, 1)

        # treat each column as a data point, each row as a variable
        if rowvar and x.shape[0] != 1:
            x = x.t()

        if ddof is None:
            if bias == 0:
                ddof = 1
            else:
                ddof = 0

        w = aweights
        if w is not None:
            if not torch.is_tensor(w):
                w = torch.tensor(w, dtype=torch.float)
            w_sum = torch.sum(w)
            avg = torch.sum(x * (w / w_sum)[:, None], 0)
        else:
            avg = torch.mean(x, 0)

        # Determine the normalization
        if w is None:
            fact = x.shape[0] - ddof
        elif ddof == 0:
            fact = w_sum
        elif aweights is None:
            fact = w_sum - ddof
        else:
            fact = w_sum - ddof * torch.sum(w * w) / w_sum

        xm = x.sub(avg.expand_as(x))

        if w is None:
            X_T = xm.t()
        else:
            X_T = torch.mm(torch.diag(w), xm).t()

        c = torch.mm(X_T, xm)
        c = c / fact

        return c.squeeze()

    def forward(self, embedding: torch.Tensor):
        """
        Calculate multivariate Gaussian distribution
        :param embedding:
        :return:
        """
        device = embedding.device

        B, C, H, W = embedding.size()
        embedding_vectors = embedding.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0)
        # mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = torch.zeros(size=(C, C, H * W), device=device)
        # cov = torch.zeros(C, C, H * W).numpy()
        I = torch.eye(C).to(device)
        # I = np.identity(C)
        for i in range(H * W):
            # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
            cov[:, :, i] = self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * I
            # cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I

        return [mean, cov]

    def fit(self, embedding: Tensor):
        return self.forward(embedding)


class Padim(torch.nn.Module):
    def __init__(self, backbone: str, layers: List[str]):
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)
        self.layers = layers
        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.features: Dict[str, List[Tensor]] = {layer: [] for layer in layers}
        self.gaussian = MultiVariateGaussian()

    def forward(self, x):
        return self.extract_features(x)

    def extract_features(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.feature_extractor(x)

        return features

    # def append_features(self, batch_features: Dict[str, Tensor]):
    def append_features(self, batch_features):
        for layer, features in batch_features.items():
            self.features[layer].append(features.detach())

        # return self.features

    def concat_features(self):
        concatenated_features: Dict[str, Tensor] = {}
        for layer, feature_list in self.features.items():
            concatenated_features[layer] = torch.cat(tensors=feature_list, dim=0)

        return concatenated_features

    @staticmethod
    def generate_embedding(features):
        def __generate_patch_embedding(x, y):
            device = x.device
            batch_x, channel_x, height_x, width_x = x.size()
            _, channel_y, height_y, width_y = y.size()
            stride = height_x // height_y
            x = F.unfold(x, kernel_size=stride, stride=stride)
            x = x.view(batch_x, channel_x, -1, height_y, width_y)
            z = torch.zeros(size=(batch_x, channel_x + channel_y, x.size(2), height_y, width_y), device=device)

            for i in range(x.size(2)):
                z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
            z = z.view(batch_x, -1, height_y * width_y)
            z = F.fold(z, kernel_size=stride, output_size=(height_x, width_x), stride=stride)

            return z

        embedding_vectors = features["layer1"]
        for layer_name in ["layer2", "layer3"]:
            embedding_vectors = __generate_patch_embedding(embedding_vectors, features[layer_name])
        return embedding_vectors

    @staticmethod
    def reduce_embedding_dimension(embedding, idx):
        # randomly select d dimension
        idx = idx.to(embedding.device)
        embedding = torch.index_select(embedding, 1, idx)

        return embedding


class AnomalyMapGenerator:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def compute_anomaly_map(self, outputs, embedding):
        # calculate distance matrix
        embedding_vectors = embedding.cpu()
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = outputs[0][:, i].cpu()
            conv_inv = np.linalg.inv(outputs[1][:, :, i].cpu())
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = (
            F.interpolate(dist_list.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=False)
            .squeeze()
            .numpy()
        )
        return score_map


def main():
    args = parse_args()

    # load model
    if args.arch == "resnet18":
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == "wide_resnet50_2":
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550

    padim = Padim(backbone=args.arch, layers=["layer1", "layer2", "layer3"])
    padim.to(device).eval()

    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    os.makedirs(os.path.join(args.save_path, "temp_%s" % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES:

        datamodule = MVTecDataModule(
            dataset_path=os.path.join(args.data_path, class_name),
            batch_size=32,
            num_workers=0,
            image_transforms=transform_x,
            mask_transforms=transform_mask,
            include_normal_images_in_val_set=True,
        )
        datamodule.setup()

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, "temp_%s" % args.arch, "train_%s.pkl" % class_name)
        for data in tqdm(datamodule.train_dataloader(), "| feature extraction | train | %s |" % class_name):
            x = data["image"].to(device)

            batch_features = padim(x)
            padim.append_features(batch_features)

        train_features = padim.concat_features()

        embedding = padim.generate_embedding(train_features)
        embedding = padim.reduce_embedding_dimension(embedding, idx)

        mean, cov = padim.gaussian.fit(embedding)

        # save learned distribution
        train_outputs = [mean, cov]
        with open(train_feature_filepath, "wb") as f:
            pickle.dump(train_outputs, f)

        print("load train set feature from: %s" % train_feature_filepath)
        with open(train_feature_filepath, "rb") as f:
            train_outputs = pickle.load(f)

        padim = Padim(backbone=args.arch, layers=["layer1", "layer2", "layer3"])
        padim.to(device).eval()

        anomaly_map_generator = AnomalyMapGenerator()

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        # extract test set features
        for data in tqdm(datamodule.test_dataloader(), "| feature extraction | test | %s |" % class_name):
            x, y, mask = data["image"].to(device), data["label"].to(device), data["mask"].to(device)

            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())

            batch_features = padim(x)
            padim.append_features(batch_features)

        test_features = padim.concat_features()

        embedding = padim.generate_embedding(test_features)
        embedding = padim.reduce_embedding_dimension(embedding, idx)

        score_map = anomaly_map_generator.compute_anomaly_map(train_outputs, embedding)

        # # calculate distance matrix
        # embedding_vectors = embedding.cpu()
        # B, C, H, W = embedding_vectors.size()
        # embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        # dist_list = []
        # for i in range(H * W):
        #     mean = train_outputs[0][:, i].cpu()
        #     conv_inv = np.linalg.inv(train_outputs[1][:, :, i].cpu())
        #     dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        #     dist_list.append(dist)
        #
        # dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
        #
        # # upsample
        # dist_list = torch.tensor(dist_list)
        # score_map = (
        #     F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode="bilinear", align_corners=False)
        #     .squeeze()
        #     .numpy()
        # )

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print("image ROCAUC: %.3f" % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label="%s img_ROCAUC: %.3f" % (class_name, img_roc_auc))

        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print("pixel ROCAUC: %.3f" % (per_pixel_rocauc))

        fig_pixel_rocauc.plot(fpr, tpr, label="%s ROCAUC: %.3f" % (class_name, per_pixel_rocauc))
        save_dir = args.save_path + "/" + f"pictures_{args.arch}"
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

    print("Average ROCAUC: %.3f" % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text("Average image ROCAUC: %.3f" % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print("Average pixel ROCUAC: %.3f" % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text("Average pixel ROCAUC: %.3f" % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, "roc_curve.png"), dpi=100)


def torch_mahalanobis(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(torch.inverse(cov), delta))
    return torch.sqrt(m)


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.0
    vmin = scores.min() * 255.0
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode="thick")
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text("Image")
        ax_img[1].imshow(gt, cmap="gray")
        ax_img[1].title.set_text("GroundTruth")
        ax = ax_img[2].imshow(heat_map, cmap="jet", norm=norm)
        ax_img[2].imshow(img, cmap="gray", interpolation="none")
        ax_img[2].imshow(heat_map, cmap="jet", alpha=0.5, interpolation="none")
        ax_img[2].title.set_text("Predicted heat map")
        ax_img[3].imshow(mask, cmap="gray")
        ax_img[3].title.set_text("Predicted mask")
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text("Segmentation result")
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            "family": "serif",
            "color": "black",
            "weight": "normal",
            "size": 8,
        }
        cb.set_label("Anomaly Score", fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + "_{}".format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.0).astype(np.uint8)

    return x


if __name__ == "__main__":
    main()
