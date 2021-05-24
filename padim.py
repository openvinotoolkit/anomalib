import argparse
import os
import pickle
import random
from random import sample
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from skimage import morphology
from skimage.segmentation import mark_boundaries
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch import Tensor
from torchvision import transforms as T
from torchvision.models import resnet18, wide_resnet50_2
from tqdm import tqdm

import datasets.mvtec as mvtec
from anomalib.datasets.mvtec import MVTecDataModule
from anomalib.datasets.utils import Denormalize
from anomalib.models.shared.feature_extractor import FeatureExtractor

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

random.seed(1024)
torch.manual_seed(1024)
if use_cuda:
    torch.cuda.manual_seed_all(1024)

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


DIMS = {"resnet18": {"t_d": 448, "d": 100}, "wide_resnet50_2": {"t_d": 1792, "d": 550}}


class Padim(torch.nn.Module):
    def __init__(self, backbone: str, layers: List[str]):
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)
        self.layers = layers
        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.features: Dict[str, List[Tensor]] = {layer: [] for layer in self.layers}
        self.gaussian = MultiVariateGaussian()
        self.dims = DIMS[backbone]
        self.idx = torch.tensor(sample(range(0, DIMS[backbone]["t_d"]), DIMS[backbone]["d"]))

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

    def clear_features(self):
        self.features: Dict[str, List[Tensor]] = {layer: [] for layer in self.layers}

    # @staticmethod
    # def generate_embedding(features, idx):
    def generate_embedding(self, features):
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

        def __reduce_embedding_dimension(embedding, idx):
            # randomly select d dimension
            idx = idx.to(embedding.device)
            embedding = torch.index_select(embedding, 1, idx)
            return embedding

        embedding_vectors = features["layer1"]
        for layer_name in ["layer2", "layer3"]:
            embedding_vectors = __generate_patch_embedding(embedding_vectors, features[layer_name])

        embedding_vectors = __reduce_embedding_dimension(embedding_vectors, self.idx)

        # # Reduce dimensions
        # self.idx = self.idx.to(embedding_vectors.device)
        # embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

        return embedding_vectors


class AnomalyMapGenerator:
    def __init__(self, image_size: int = 224, alpha: float = 0.4, gamma: int = 0, sigma: int = 4, kernel_size: int = 4):
        self.image_size = image_size
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.gamma = gamma

    @staticmethod
    def compute_distance(embedding, outputs):
        embedding_vectors = embedding.cpu()
        batch, channel, height, width = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(batch, channel, height * width).numpy()
        dist_list = []
        for i in range(height * width):
            mean = outputs[0][:, i].cpu()
            conv_inv = np.linalg.inv(outputs[1][:, :, i].cpu())
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(batch, height, width)
        dist_list = torch.tensor(dist_list)

        return dist_list

    def upsample(self, dist_list):
        score_map = (
            F.interpolate(dist_list.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=False)
            .squeeze()
            .numpy()
        )
        return score_map

    def smooth_anomaly_map(self, anomaly_map):
        if isinstance(anomaly_map, Tensor):
            anomaly_map = anomaly_map.cpu().numpy()

        # apply gaussian smoothing on the score map
        for i in range(anomaly_map.shape[0]):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=self.sigma)

        return anomaly_map

    @staticmethod
    def normalize(anomaly_map):
        # Normalization
        max_score = anomaly_map.max()
        min_score = anomaly_map.min()
        scores = (anomaly_map - min_score) / (max_score - min_score)
        return scores

    def compute_anomaly_map(self, embedding, stats):
        score_map = self.compute_distance(embedding, stats)
        score_map = self.upsample(score_map)
        score_map = self.smooth_anomaly_map(score_map)
        score_map = self.normalize(score_map)

        return score_map

    @staticmethod
    def compute_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
        anomaly_map = anomaly_map * 255
        anomaly_map = anomaly_map.astype(np.uint8)

        heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        return heatmap

    def apply_heatmap_on_image(self, anomaly_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        heatmap = self.compute_heatmap(anomaly_map)
        heatmap_on_image = cv2.addWeighted(heatmap, self.alpha, image, self.beta, self.gamma)
        heatmap_on_image = cv2.cvtColor(heatmap_on_image, cv2.COLOR_BGR2RGB)
        return heatmap_on_image

    @staticmethod
    def compute_adaptive_threshold(true_masks, anomaly_map):
        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(true_masks.flatten(), anomaly_map.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        return threshold

    def compute_mask(self, anomaly_map: np.ndarray, threshold: float) -> np.ndarray:
        mask = np.zeros_like(anomaly_map).astype(np.uint8)
        mask[anomaly_map > threshold] = 1

        kernel = morphology.disk(self.kernel_size)
        mask = morphology.opening(mask, kernel)

        mask *= 255

        return mask


class Visualizer:
    def __init__(self, num_rows: int, num_cols: int, figure_size: Tuple[int, int]):
        self.figure_index: int = 0

        self.figure, self.axis = plt.subplots(num_rows, num_cols, figsize=figure_size)
        self.figure.subplots_adjust(right=0.9)

        for axis in self.axis:
            axis.axes.xaxis.set_visible(False)
            axis.axes.yaxis.set_visible(False)

    def add_image(self, index: int, image: np.ndarray, title: str, cmap: Optional[str] = None):
        self.axis[index].imshow(image, cmap)
        self.axis[index].title.set_text(title)

    def show(self):
        self.figure.show()

    def save(self, filename: str):
        self.figure.savefig(filename, dpi=100)

    def close(self):
        plt.close(self.figure)


def main():
    args = parse_args()

    padim = Padim(backbone=args.arch, layers=["layer1", "layer2", "layer3"])
    padim.to(device).eval()

    idx = torch.tensor(sample(range(0, DIMS[args.arch]["t_d"]), DIMS[args.arch]["d"]))

    os.makedirs(os.path.join(args.save_path, "temp_%s" % args.arch), exist_ok=True)

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
        train_feature_filepath = os.path.join(args.save_path, "temp_%s" % args.arch, "train_%s.pkl" % class_name)

        # TRAINING
        # extract train set features
        for data in tqdm(datamodule.train_dataloader(), "| feature extraction | train | %s |" % class_name):
            x = data["image"].to(device)

            batch_features = padim(x)
            padim.append_features(batch_features)

        train_features = padim.concat_features()
        embedding = padim.generate_embedding(train_features)
        mean, cov = padim.gaussian.fit(embedding)

        # save learned distribution
        train_outputs = [mean, cov]
        with open(train_feature_filepath, "wb") as f:
            pickle.dump(train_outputs, f)

        print("load train set feature from: %s" % train_feature_filepath)
        with open(train_feature_filepath, "rb") as f:
            train_outputs = pickle.load(f)

        # TEST
        padim.clear_features()

        anomaly_map_generator = AnomalyMapGenerator()

        true_labels = []
        true_masks = []
        test_imgs = []
        image_filenames = []

        # extract test set features
        for data in tqdm(datamodule.test_dataloader(), "| feature extraction | test | %s |" % class_name):
            x, y, mask = data["image"].to(device), data["label"].to(device), data["mask"].to(device)

            image_filenames.extend(data["image_path"])
            test_imgs.extend(x)
            # test_imgs.extend(x.cpu().detach().numpy())
            true_labels.extend(y.cpu().detach().numpy())
            true_masks.extend(mask.cpu().detach().numpy().squeeze())

            batch_features = padim(x)
            padim.append_features(batch_features)

        test_features = padim.concat_features()
        embedding = padim.generate_embedding(test_features)
        anomaly_maps = anomaly_map_generator.compute_anomaly_map(embedding, train_outputs)

        # Compute performance.
        pred_labels = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(axis=1)
        true_labels = np.asarray(true_labels)
        true_masks = np.asarray(true_masks)

        image_roc_auc = roc_auc_score(true_labels, pred_labels)
        pixel_roc_auc = roc_auc_score(true_masks.flatten(), anomaly_maps.flatten())
        print(f"Image-Level ROC AUC: {image_roc_auc:.3f}\n Pixel-Level ROC AUC: {pixel_roc_auc:.3f}")

        save_dir = args.save_path + "/" + f"pictures_{args.arch}"
        os.makedirs(save_dir, exist_ok=True)
        threshold = anomaly_map_generator.compute_adaptive_threshold(true_masks, anomaly_maps)

        for i, (filename, image, true_mask, anomaly_map) in enumerate(
            zip(image_filenames, test_imgs, true_masks, anomaly_maps)
        ):
            image = Denormalize()(image)

            heat_map = anomaly_map_generator.apply_heatmap_on_image(anomaly_map, image)
            pred_mask = anomaly_map_generator.compute_mask(anomaly_map=anomaly_map, threshold=threshold)
            vis_img = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")

            visualizer = Visualizer(num_rows=1, num_cols=5, figure_size=(12, 3))
            visualizer.add_image(index=0, image=image, title="Image")
            visualizer.add_image(index=1, image=true_mask, cmap="gray", title="Ground Truth")
            visualizer.add_image(index=2, image=heat_map, title="Predicted Heat Map")
            visualizer.add_image(index=3, image=pred_mask, cmap="gray", title="Predicted Mask")
            visualizer.add_image(index=4, image=vis_img, title="Segmentation Result")
            visualizer.save(os.path.join(save_dir, class_name + "_{}".format(i)))
            visualizer.close()


def torch_mahalanobis(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(torch.inverse(cov), delta))
    return torch.sqrt(m)


if __name__ == "__main__":
    main()
