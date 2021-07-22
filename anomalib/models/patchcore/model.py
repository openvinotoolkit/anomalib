import argparse
import glob
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from numpy.lib.arraysetops import isin
from omegaconf import ListConfig
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import wide_resnet50_2

from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.core.utils.anomaly_map_generator import BaseAnomalyMapGenerator
from anomalib.datasets.utils import Denormalize
from anomalib.models.patchcore.sampling_methods.kcenter_greedy import kCenterGreedy

# def copy_files(src, dst, ignores=[]):
#     src_files = os.listdir(src)
#     for file_name in src_files:
#         ignore_check = [True for i in ignores if i in file_name]
#         if ignore_check:
#             continue
#         full_file_name = os.path.join(src, file_name)
#         if os.path.isfile(full_file_name):
#             shutil.copy(full_file_name, os.path.join(dst, file_name))
#         if os.path.isdir(full_file_name):
#             os.makedirs(os.path.join(dst, file_name), exist_ok=True)
#             copy_files(full_file_name, os.path.join(dst, file_name), ignores)


def prep_dirs(root, category):
    # make embeddings dir
    # embeddings_path = os.path.join(root, 'embeddings')
    embeddings_path = os.path.join("./", "embeddings", category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, "sample")
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, "src")
    os.makedirs(source_code_save_path, exist_ok=True)
    # copy_files(
    #     "./", source_code_save_path, [".git", ".vscode", "__pycache__", "logs", "README", "samples", "LICENSE"]
    # )  # copy source code
    return embeddings_path, sample_path, source_code_save_path


def concat_layer_embedding(embedding: Tensor, layer_embedding: Tensor) -> Tensor:
    """
    Generate patch embedding via pixel patches. A quote from Section IIIA from the paper:

    "As activation maps have a lower resolution  than  the  input  image,
    many  pixels  have  the  same embeddings  and  then  form  pixel  patches
    with  no  overlap  in the  original  image  resolution.  Hence,  an  input
    image  can  be divided  in  a  grid  of (i,j) ∈ [1,W] × [1,H] positions  where
    WxH is  the  resolution  of  the  largest  activation  map  used  to
    generate embeddings."

    :param embedding: Embedding vector from the earlier layers
    :param layer_features: Feature map from the subsequent layer.
    :return:
    """
    device = embedding.device
    batch_x, channel_x, height_x, width_x = embedding.size()
    _, channel_y, height_y, width_y = layer_embedding.size()
    stride = height_x // height_y
    embedding = F.unfold(embedding, kernel_size=stride, stride=stride)
    embedding = embedding.view(batch_x, channel_x, -1, height_y, width_y)
    updated_embedding = torch.zeros(
        size=(batch_x, channel_x + channel_y, embedding.size(2), height_y, width_y), device=device
    )

    for i in range(embedding.size(2)):
        updated_embedding[:, :, i, :, :] = torch.cat((embedding[:, :, i, :, :], layer_embedding), 1)
    updated_embedding = updated_embedding.view(batch_x, -1, height_y * width_y)
    updated_embedding = F.fold(updated_embedding, kernel_size=stride, output_size=(height_x, width_x), stride=stride)

    return updated_embedding


def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


# def reshape_embedding(embedding):
#     embedding_list = []
#     for k in range(embedding.shape[0]):
#         for i in range(embedding.shape[2]):
#             for j in range(embedding.shape[3]):
#                 embedding_list.append(embedding[k, :, i, j])
#     return embedding_list


def reshape_embedding(embedding: Tensor) -> Tensor:
    """
    Reshapes Embedding to the following format:
    [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

    Args:
        embedding (Tensor): Embedding tensor extracted from CNN features.

    Returns:
        Tensor: Reshaped embedding tensor.
    """
    # [batch, embedding, patch, patch] -> [batch*patch*patch, embedding]
    embedding_size = embedding.size(1)
    embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
    return embedding


# imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cal_confusion_matrix(y_true, y_pred_no_thresh, thresh, img_path_list):
    pred_thresh = []
    false_n = []
    false_p = []
    for i in range(len(y_pred_no_thresh)):
        if y_pred_no_thresh[i] > thresh:
            pred_thresh.append(1)
            if y_true[i] == 0:
                false_p.append(img_path_list[i])
        else:
            pred_thresh.append(0)
            if y_true[i] == 1:
                false_n.append(img_path_list[i])

    cm = confusion_matrix(y_true, pred_thresh)
    print(cm)
    print("false positive")
    print(false_p)
    print("false negative")
    print(false_n)


class AnomalyMapGenerator(BaseAnomalyMapGenerator):
    """
    Generate Anomaly Heatmap
    """

    def __init__(
        self,
        batch_size: int = 1,
        image_size: Union[ListConfig, Tuple] = (256, 256),
        alpha: float = 0.4,
        gamma: int = 0,
        sigma: int = 4,
    ):
        super().__init__(alpha=alpha, gamma=gamma, sigma=sigma)
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.batch_size = batch_size

    def compute_anomaly_map(self, score_patches: np.ndarray) -> np.ndarray:
        """
        Pixel Level Anomaly Heatmap

        Args:
            score_patches (np.ndarray): [description]
        """
        anomaly_map = score_patches[:, 0].reshape((28, 28))
        anomaly_map = cv2.resize(anomaly_map, self.image_size)
        anomaly_map = gaussian_filter(anomaly_map, sigma=self.sigma)

        return anomaly_map

    @staticmethod
    def compute_anomaly_score(score_patches: np.ndarray) -> np.ndarray:
        """
        Compute Image-Level Anomaly Score

        Args:
            score_patches (np.ndarray): [description]
        """
        N_b = score_patches[np.argmax(score_patches[:, 0])]
        w = 1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b)))
        score = w * max(score_patches[:, 0])
        return score

    def __call__(self, score_patches: np.ndarray) -> np.ndarray:
        return self.compute_anomaly_map(score_patches)


class PatchcoreModel(torch.nn.Module):
    """
    Padim Module
    """

    def __init__(self, backbone: str, layers: List[str]):
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)
        self.layers = layers
        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.nn_search = NearestNeighbors(n_neighbors=9)

    def forward(self, input_tensor: Tensor) -> np.ndarray:
        """Forward-pass image-batch (N, C, H, W) into model to extract features.

        Args:
            input_tensor: Image-batch (N, C, H, W)
            input_tensor: Tensor:

        Returns:
            Features from single/multiple layers.

        Examples:

        >>> x = torch.randn(32, 3, 224, 224)
        >>> features = self.extract_features(input_tensor)
        >>> features.keys()
        dict_keys(['layer1', 'layer2', 'layer3'])

        >>> [v.shape for v in features.values()]
        [torch.Size([32, 64, 56, 56]),
         torch.Size([32, 128, 28, 28]),
         torch.Size([32, 256, 14, 14])]
        """
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)
        # features = [self.feature_pooler(feature) for feature in features.values()]

        return embedding

    # def append_features(self, features: Dict[str, Tensor]) -> List[Tensor]:
    #     # def append_features(self, outputs: List[Dict[str, Any]]) -> Dict[str, List[Tensor]]:
    #     """append_features from each batch to concatenate

    #     Args:
    #             features description]
    #             features: List[Dict[str:Tensor]]:

    #     Returns:
    #             description]

    #     """
    #     pool = torch.nn.AvgPool2d(3, 1, 1)
    #     appended_features = [pool(feature) for feature in features]
    #     # embeddings = [pool(feature) for feature in features.values()]

    #     return appended_features
    #
    # @staticmethod
    # def concat_features(features: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
    #     """Concatenate batch features to form one big feauture matrix.

    #     Args:
    #                     features: Features from batches.
    #                     features: Dict[str:
    #                     List[Tensor]]:

    #     Returns:
    #                     Concatenated feature map.

    #     """
    #     concatenated_features: Dict[str, Tensor] = {}
    #     for layer, feature_list in features.items():
    #         concatenated_features[layer] = torch.cat(tensors=feature_list, dim=0)

    #     return concatenated_features

    def generate_embedding(self, features: Dict[str, Tensor]) -> np.ndarray:
        """Generate embedding from hierarchical feature map

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: Dict[str:Tensor]:

        Returns:
                Embedding vector

        """

        layer_embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embeddings = concat_layer_embedding(layer_embeddings, features[layer])

        embedding = reshape_embedding(layer_embeddings).cpu().numpy()
        return embedding

    @staticmethod
    def subsample_embedding(embedding: np.ndarray, sampling_ratio: float) -> np.ndarray:
        """
        Subsample embedding based on coreset sampling

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio

        Returns:
            np.ndarray: Subsampled embedding whose dimensionality is reduced.
        """

        # Random projection
        random_projector = SparseRandomProjection(n_components="auto", eps=0.9)  # 'auto' => Johnson-Lindenstrauss lemma
        random_projector.fit(embedding)
        # Coreset Subsampling
        selector = kCenterGreedy(embedding, 0, 0)
        selected_idx = selector.select_batch(
            model=random_projector,
            already_selected=[],
            N=int(embedding.shape[0] * sampling_ratio),
        )
        embedding_coreset = embedding[selected_idx]
        return embedding_coreset


class PatchcoreLightning(pl.LightningModule):
    def __init__(self, hparams):
        super(PatchcoreLightning, self).__init__()

        self.save_hyperparameters(hparams)

        # self.init_features()

        # def hook_t(module, input, output):
        #     self.features.append(output)

        # self.model = wide_resnet50_2(pretrained=True)
        self._model = PatchcoreModel(backbone=hparams.model.backbone, layers=hparams.model.layers).eval()
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=hparams.dataset.crop_size)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # self.model.layer2[-1].register_forward_hook(hook_t)
        # self.model.layer3[-1].register_forward_hook(hook_t)

        # self.init_results_list()

        self.automatic_optimization = False

        # self.inv_normalize = transforms.Normalize(
        #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255], std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        # )

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []

    # def init_features(self):
    #     self.features = []

    # def forward(self, x_t):
    #     self.init_features()
    #     _ = self.model(x_t)
    #     return self.features

    def save_anomaly_map(self, hm_on_img, gt_img, file_name, x_type):

        # save images
        # cv2.imwrite(os.path.join(self.hparams.project.path, "images", f"{x_type}_{file_name}.jpg"), input_img)
        # cv2.imwrite(
        #     os.path.join(self.hparams.project.path, "images", f"{x_type}_{file_name}_amap.jpg"), anomaly_map_norm_hm
        # )
        cv2.imwrite(
            os.path.join(self.hparams.project.path, "images", f"{x_type}_{file_name}_amap_on_img.jpg"), hm_on_img
        )
        cv2.imwrite(os.path.join(self.hparams.project.path, "images", f"{x_type}_{file_name}_gt.jpg"), gt_img)

    # def train_dataloader(self):
    #     image_datasets = MVTecDataset(
    #         root=os.path.join(self.hparams.dataset.path, self.hparams.dataset.category),
    #         transform=self.data_transforms,
    #         gt_transform=self.gt_transforms,
    #         phase="train",
    #     )
    #     train_loader = DataLoader(
    #         image_datasets, batch_size=self.hparams.dataset.batch_size, shuffle=True, num_workers=0
    #     )  # , pin_memory=True)
    #     return train_loader

    # def test_dataloader(self):
    #     test_datasets = MVTecDataset(
    #         root=os.path.join(self.hparams.dataset.path, self.hparams.dataset.category),
    #         transform=self.data_transforms,
    #         gt_transform=self.gt_transforms,
    #         phase="test",
    #     )
    #     test_loader = DataLoader(
    #         test_datasets, batch_size=1, shuffle=False, num_workers=0
    #     )  # , pin_memory=True) # only work on batch_size=1, now.
    #     return test_loader

    def configure_optimizers(self):
        return None

    # def on_train_start(self):
    #     self.model.eval()  # to stop running_var move (maybe not critical)
    # self.embedding_list = []

    def on_test_start(self):
        self.init_results_list()

    def training_step(self, batch, batch_idx):  # save locally aware patch features
        # images = batch["image"]
        # features = self(batch["image"])
        # features2 = self._model(batch["image"])
        self._model.eval()
        embedding = self._model(batch["image"])
        # features = self._model(batch["image"])
        # features = self(batch["image"])

        # embeddings = []
        # for feature in features:
        #     m = torch.nn.AvgPool2d(3, 1, 1)
        #     embeddings.append(m(feature))

        # embeddings = self._model.append_features(features)
        # embeddings = self._model(batch["image"])
        # embedding = embedding_concat(embeddings[0], embeddings[1])
        # # TODO: self._model.generate_embedding
        # embedding = concat_layer_embedding(features[0], features[1])
        # embedding = reshape_embedding(embedding)
        # embedding = self._model.generate_embedding(features)

        # Embedding is used coreset subsampling which requires numpy arrays
        # embedding = embedding.cpu().numpy()

        return {"embedding": embedding}

    def training_epoch_end(self, outputs):
        # total_embeddings = np.array(self.embedding_list)
        # total_embeddings = total_embeddings.cpu().numpy()
        embedding = np.vstack([output["embedding"] for output in outputs])
        sampling_ratio = self.hparams.model.coreset_sampling_ratio
        embedding = self._model.subsample_embedding(embedding, sampling_ratio)
        # Random projection
        # random_projector = SparseRandomProjection(n_components="auto", eps=0.9)  # 'auto' => Johnson-Lindenstrauss lemma
        # random_projector.fit(embedding)
        # # Coreset Subsampling
        # selector = kCenterGreedy(embedding, 0, 0)
        # selected_idx = selector.select_batch(
        #     model=random_projector,
        #     already_selected=[],
        #     N=int(embedding.shape[0] * self.hparams.model.coreset_sampling_ratio),
        # )
        # embedding_coreset = embedding[selected_idx]

        # print("initial embedding size : ", embedding.shape)
        # print("final embedding size : ", embedding.shape)
        with open(os.path.join(self.hparams.project.path, "weights", "embedding.pickle"), "wb") as file:
            pickle.dump(embedding, file)

    def test_step(self, batch, batch_idx):  # Nearest Neighbour Search
        with open(os.path.join(self.hparams.project.path, "weights", "embedding.pickle"), "rb") as file:
            memory_bank = pickle.load(file)
        # x, gt, label, file_name, x_type = batch
        filenames, images, labels, masks = batch["image_path"], batch["image"], batch["label"], batch["mask"]

        filenames = Path(filenames[0])
        filename = filenames.stem
        category = filenames.parent.name
        # features = self._model(images)

        # # extract embedding
        # features = self(images)
        # # features = self._model(images)
        # # embeddings = []
        # # for feature in features:
        # #     m = torch.nn.AvgPool2d(3, 1, 1)
        # #     embeddings.append(m(feature))
        # embeddings = self._model.append_features(features)
        # # embeddings = self._model(images)
        # # embedding_ = embedding_concat(embeddings[0], embeddings[1])
        # embedding = concat_layer_embedding(embeddings[0], embeddings[1])
        # # embedding_test = np.array(reshape_embedding(np.array(embedding_.cpu())))
        # embedding_test = reshape_embedding(embedding).cpu().numpy()

        # features = self._model(images)
        # embedding_test = self._model.generate_embedding(features)
        embedding = self._model(images)

        # nn_classifier = NearestNeighbors(n_neighbors=self.hparams.model.num_neighbors).fit(normal_embedding)
        nn_search = self._model.nn_search.fit(memory_bank)
        patch_scores, _ = nn_search.kneighbors(embedding)

        # Pixel Level Anomaly Heatmap
        # anomaly_map = score_patches[:, 0].reshape((28, 28))
        # anomaly_map_resized = cv2.resize(anomaly_map, tuple(self.hparams.dataset.crop_size))
        # anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)

        anomaly_map = self.anomaly_map_generator.compute_anomaly_map(patch_scores)
        score = self.anomaly_map_generator.compute_anomaly_score(patch_scores)
        # # Image-Level Anomaly Score
        # N_b = score_patches[np.argmax(score_patches[:, 0])]
        # w = 1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b)))
        # score = w * max(score_patches[:, 0])  # Image-level score

        gt_np = masks.cpu().numpy()[0, 0].astype(int)

        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map.ravel())
        self.gt_list_img_lvl.append(labels.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(filename)
        # save images
        # images = self.inv_normalize(images)
        images = Denormalize()(images)
        input_x = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        # input_x = cv2.cvtColor(images.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)

        anomaly_map_norm = min_max_norm(anomaly_map)
        heatmap = cvt2heatmap(anomaly_map_norm * 255)
        hm_on_img = heatmap_on_image(heatmap, input_x)

        self.save_anomaly_map(hm_on_img, gt_np * 255, filename, category)
        # return {
        #     "filenames": filenames,
        #     "images": images,
        #     "features": features,
        #     "true_labels": labels.cpu().numpy(),
        #     "true_masks": masks.squeeze().cpu().numpy(),
        # }

    def test_epoch_end(self, outputs):
        print("Total pixel-level auc-roc score :")
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print("test_epoch_end")
        values = {"pixel_auc": pixel_auc, "img_auc": img_auc}
        self.log_dict(values)
        # anomaly_list = []
        # normal_list = []
        # for i in range(len(self.gt_list_img_lvl)):
        #     if self.gt_list_img_lvl[i] == 1:
        #         anomaly_list.append(self.pred_list_img_lvl[i])
        #     else:
        #         normal_list.append(self.pred_list_img_lvl[i])

        # # thresholding
        # # cal_confusion_matrix(self.gt_list_img_lvl, self.pred_list_img_lvl, img_path_list = self.img_path_list, thresh = 0.00097)
        # # print()
        # with open(args.project_root_path + r'/results.txt', 'a') as f:
        #     f.write(args.category + ' : ' + str(values) + '\n')


#
#
# def get_args():
#     parser = argparse.ArgumentParser(description="ANOMALYDETECTION")
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


# if __name__ == "__main__":

#     args = get_args()

#     trainer = pl.Trainer.from_argparse_args(
#         args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=1
#     )  # , check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
#     model = PatchcoreLightning(hparams=args)
#     if args.phase == "train":
#         trainer.fit(model)
#         trainer.test(model)
#     elif args.phase == "test":
#         trainer.test(model)
