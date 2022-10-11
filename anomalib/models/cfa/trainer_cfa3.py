import argparse
import random
import warnings
from typing import Optional

import datasets.mvtec as mvtec
import torch
import torch.optim as optim
from cnn.efficientnet import EfficientNet as effnet
from cnn.resnet import resnet18 as res18
from cnn.resnet import wide_resnet50_2 as wrn50_2
from cnn.vgg import vgg19_bn as vgg19
from datasets.mvtec import MVTecDataset
from torch.utils.data import DataLoader
from utils.cfa3 import *
from utils.metric import *
from utils.visualizer import *

warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def parse_args():
    parser = argparse.ArgumentParser("CFA configuration")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str, default="./mvtec_result")
    parser.add_argument("--Rd", type=bool, default=False)
    parser.add_argument("--backbone", type=str, choices=["res18", "wrn50_2", "effnet-b5", "vgg19"], default="wrn50_2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--size", type=int, choices=[224, 256], default=224)
    parser.add_argument("--gamma_c", type=int, default=1)
    parser.add_argument("--gamma_d", type=int, default=1)

    parser.add_argument("--class_name", type=str, default="all")

    return parser.parse_args()


def run():
    seed = 1024
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    args = parse_args()
    class_names = mvtec.CLASS_NAMES if args.class_name == "all" else [args.class_name]

    total_roc_auc = []
    total_pixel_roc_auc = []
    total_pixel_pro_auc = []

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    for class_name in class_names:
        best_img_roc = -1
        best_pxl_roc = -1
        best_pxl_pro = -1
        print(" ")
        print("%s | newly initialized..." % class_name)

        train_dataset = MVTecDataset(
            dataset_path=args.data_path,
            class_name=class_name,
            resize=256,
            cropsize=args.size,
            is_train=True,
            wild_ver=args.Rd,
        )

        test_dataset = MVTecDataset(
            dataset_path=args.data_path,
            class_name=class_name,
            resize=256,
            cropsize=args.size,
            is_train=False,
            wild_ver=args.Rd,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=4,
            pin_memory=True,
        )

        feature_extractor = get_feature_extractor(args.backbone)
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()

        model = CfaModel(feature_extractor, train_loader, args.backbone, args.gamma_c, args.gamma_d, device)
        model = model.to(device)

        optimizer = optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=5e-4, amsgrad=True)

        for epoch in tqdm(range(args.epochs), "%s -->" % (class_name)):
            r"TEST PHASE"

            test_imgs = list()
            gt_mask_list = list()
            gt_list = list()
            heatmaps = None

            model.train()
            for (x, _, _) in train_loader:
                optimizer.zero_grad()
                x = x.to(device)
                with torch.no_grad():
                    p = feature_extractor(x)

                loss = model(p)
                loss.backward()
                optimizer.step()

            model.eval()
            for x, y, mask in test_loader:
                test_imgs.extend(x.cpu().detach().numpy())
                gt_list.extend(y.cpu().detach().numpy())
                gt_mask_list.extend(mask.cpu().detach().numpy())

                x = x.to(device)
                with torch.no_grad():
                    p = feature_extractor(x)

                score = model(p)
                heatmap = score.cpu().detach()
                anomaly_map = heatmap.clone()
                heatmap = torch.mean(heatmap, dim=1)
                heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap

            heatmaps = upsample(heatmaps, size=x.size(2), mode="bilinear")
            heatmaps = gaussian_smooth(heatmaps, sigma=4)

            gt_mask = np.asarray(gt_mask_list)
            scores = rescale(heatmaps)

            scores = scores
            threshold = get_threshold(gt_mask, scores)

            r"Image-level AUROC"
            fpr, tpr, img_roc_auc = cal_img_roc(scores, gt_list)
            best_img_roc = img_roc_auc if img_roc_auc > best_img_roc else best_img_roc

            fig_img_rocauc.plot(fpr, tpr, label="%s img_ROCAUC: %.3f" % (class_name, img_roc_auc))

            r"Pixel-level AUROC"
            fpr, tpr, per_pixel_rocauc = cal_pxl_roc(gt_mask, scores)
            best_pxl_roc = per_pixel_rocauc if per_pixel_rocauc > best_pxl_roc else best_pxl_roc

            r"Pixel-level AUPRO"
            per_pixel_proauc = cal_pxl_pro(gt_mask, scores)
            best_pxl_pro = per_pixel_proauc if per_pixel_proauc > best_pxl_pro else best_pxl_pro

            print("[%d / %d]image ROCAUC: %.3f | best: %.3f" % (epoch, args.epochs, img_roc_auc, best_img_roc))
            print("[%d / %d]pixel ROCAUC: %.3f | best: %.3f" % (epoch, args.epochs, per_pixel_rocauc, best_pxl_roc))
            print("[%d / %d]pixel PROAUC: %.3f | best: %.3f" % (epoch, args.epochs, per_pixel_proauc, best_pxl_pro))

        print("image ROCAUC: %.3f" % (best_img_roc))
        print("pixel ROCAUC: %.3f" % (best_pxl_roc))
        print("pixel ROCAUC: %.3f" % (best_pxl_pro))

        total_roc_auc.append(best_img_roc)
        total_pixel_roc_auc.append(best_pxl_roc)
        total_pixel_pro_auc.append(best_pxl_pro)

        fig_pixel_rocauc.plot(fpr, tpr, label="%s ROCAUC: %.3f" % (class_name, per_pixel_rocauc))
        save_dir = args.save_path + "/" + f"pictures_{args.backbone}"
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

    print("Average ROCAUC: %.3f" % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text("Average image ROCAUC: %.3f" % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print("Average pixel ROCUAC: %.3f" % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text("Average pixel ROCAUC: %.3f" % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    print("Average pixel PROUAC: %.3f" % np.mean(total_pixel_pro_auc))

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, "roc_curve.png"), dpi=100)

def get_feature_extractor(backbone: str, device: Optional[torch.device] = None):
    if backbone == "wrn50_2":
        feature_extractor = wrn50_2(pretrained=True, progress=True)
    elif backbone == "res18":
        feature_extractor = res18(pretrained=True, progress=True)
    elif backbone == "effnet-b5":
        feature_extractor = effnet.from_pretrained("efficientnet-b5")
    elif backbone == "vgg19":
        feature_extractor = vgg19(pretrained=True, progress=True)

    if device is not None:
        feature_extractor = feature_extractor.to(device)

    return feature_extractor


if __name__ == "__main__":
    run()
