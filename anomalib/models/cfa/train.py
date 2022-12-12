import argparse
import random
import warnings

import datasets.mvtec as mvtec
import torch
import torch.optim as optim
from cnn.efficientnet import EfficientNet as effnet
from cnn.resnet import resnet18, wide_resnet50_2
from cnn.vgg import vgg19_bn
from datasets.mvtec import MVTecDataset
from torch.utils.data import DataLoader
from torch_model import CfaModel
from utils.cfa import *
from utils.metric import *
from utils.visualizer import *

warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def parse_args():
    parser = argparse.ArgumentParser("CFA configuration")
    parser.add_argument("--data_path", type=str, default="/home/sakcay/projects/anomalib/datasets/MVTec/")
    parser.add_argument("--save_path", type=str, default="./mvtec_result")
    parser.add_argument("--Rd", type=bool, default=False)
    parser.add_argument(
        "--cnn",
        type=str,
        choices=["resnet18", "wide_resnet50_2", "efficientnet_b5", "vgg19_bn"],
        default="wide_resnet50_2",
    )
    parser.add_argument("--resize", type=int, choices=[224, 256], default=224)
    parser.add_argument("--size", type=int, choices=[224, 256], default=224)
    parser.add_argument("--gamma_c", type=int, default=1)
    parser.add_argument("--gamma_d", type=int, default=1)

    parser.add_argument("--class_name", type=str, default="zipper")

    return parser.parse_args()


def run():
    args = parse_args()
    class_names = [random.choice(mvtec.CLASS_NAMES)]
    print(f"Class name: {class_names[0]}")

    seed = 1024
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    total_roc_auc = []
    total_pixel_roc_auc = []
    total_pixel_pro_auc = []

    # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # fig_img_rocauc = ax[0]
    # fig_pixel_rocauc = ax[1]

    for class_name in class_names:
        best_img_roc1 = -1
        best_img_roc2 = -1
        best_pxl_roc1 = -1
        best_pxl_roc2 = -1
        best_pxl_pro1 = -1
        best_pxl_pro2 = -1
        print(" ")
        print("%s | newly initialized..." % class_name)

        train_dataset = MVTecDataset(
            dataset_path=args.data_path,
            class_name=class_name,
            resize=args.resize,
            cropsize=args.size,
            is_train=True,
            wild_ver=args.Rd,
        )

        test_dataset = MVTecDataset(
            dataset_path=args.data_path,
            class_name=class_name,
            resize=args.resize,
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

        if args.cnn == "wide_resnet50_2":
            feature_extractor1 = wide_resnet50_2(pretrained=True, progress=True)
        elif args.cnn == "resnet18":
            feature_extractor1 = resnet18(pretrained=True, progress=True)
        elif args.cnn == "efficientnet_b5":
            feature_extractor1 = effnet.from_pretrained("efficientnet-b5")
        elif args.cnn == "vgg19_bn":
            feature_extractor1 = vgg19_bn(pretrained=True, progress=True)

        feature_extractor1 = feature_extractor1.to(device)
        feature_extractor1.eval()

        cfa1 = DSVDD(feature_extractor1, train_loader, args.cnn, args.gamma_c, args.gamma_d, device)
        cfa2 = CfaModel(feature_extractor1, train_loader, args.cnn, args.gamma_c, args.gamma_d, device)

        cfa1 = cfa1.to(device)
        cfa2 = cfa2.to(device)

        epochs = 1
        params1 = [{"params": cfa1.parameters()}]
        params2 = [{"params": cfa2.parameters()}]
        optimizer1 = optim.AdamW(params=params1, lr=1e-3, weight_decay=5e-4, amsgrad=True)
        optimizer2 = optim.AdamW(params=params2, lr=1e-3, weight_decay=5e-4, amsgrad=True)

        for epoch in tqdm(range(epochs), "%s -->" % (class_name)):
            r"TEST PHASE"

            test_imgs = list()
            gt_mask_list = list()
            gt_list = list()
            heatmaps1 = None
            heatmaps2 = None

            cfa1.train()
            cfa2.train()
            for (batch, _, _) in train_loader:
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                features1 = feature_extractor1(batch.to(device))

                loss1, _ = cfa1(features1)
                loss2 = cfa2(features1)
                loss1.backward(retain_graph=True)
                loss2.backward(retain_graph=True)
                optimizer1.step()
                optimizer2.step()
                print(f"loss1: {loss1}, loss2: {loss2}")

            cfa1.eval()
            cfa2.eval()
            for batch, y, mask in test_loader:
                test_imgs.extend(batch.cpu().detach().numpy())
                gt_list.extend(y.cpu().detach().numpy())
                gt_mask_list.extend(mask.cpu().detach().numpy())

                features1 = feature_extractor1(batch.to(device))
                _, score1 = cfa1(features1)
                # score2 = cfa2(features1)
                heatmap1 = score1.cpu().detach()
                heatmap2 = cfa2(features1)
                # heatmap2 = score2.cpu().detach()
                heatmap1 = torch.mean(heatmap1, dim=1)
                # heatmap2 = torch.mean(heatmap2, dim=1)
                heatmaps1 = torch.cat((heatmaps1, heatmap1), dim=0) if heatmaps1 != None else heatmap1
                heatmaps2 = torch.cat((heatmaps2, heatmap2), dim=0) if heatmaps2 != None else heatmap2

            heatmaps1 = upsample(heatmaps1, size=batch.size(2), mode="bilinear")
            # heatmaps2 = upsample(heatmaps2, size=batch.size(2), mode="bilinear")
            heatmaps1 = gaussian_smooth(heatmaps1, sigma=4)
            # heatmaps2 = gaussian_smooth(heatmaps2, sigma=4)

            gt_mask = np.asarray(gt_mask_list)
            scores1 = rescale(heatmaps1)
            scores2 = rescale(heatmaps2.squeeze().cpu().numpy())

            # threshold1 = get_threshold(gt_mask, scores1)
            # threshold2 = get_threshold(gt_mask, scores2)

            r"Image-level AUROC"
            fpr1, tpr1, img_roc_auc1 = cal_img_roc(scores1, gt_list)
            fpr2, tpr2, img_roc_auc2 = cal_img_roc(scores2, gt_list)
            best_img_roc1 = img_roc_auc1 if img_roc_auc1 > best_img_roc1 else best_img_roc1
            best_img_roc2 = img_roc_auc2 if img_roc_auc2 > best_img_roc2 else best_img_roc2

            # fig_img_rocauc.plot(fpr1, tpr1, label="%s img_ROCAUC: %.3f" % (class_name, img_roc_auc1))

            r"Pixel-level AUROC"
            fpr1, tpr1, per_pixel_rocauc1 = cal_pxl_roc(gt_mask, scores1)
            fpr2, tpr2, per_pixel_rocauc2 = cal_pxl_roc(gt_mask, scores2)
            best_pxl_roc1 = per_pixel_rocauc1 if per_pixel_rocauc1 > best_pxl_roc1 else best_pxl_roc1
            best_pxl_roc2 = per_pixel_rocauc2 if per_pixel_rocauc2 > best_pxl_roc2 else best_pxl_roc2

            r"Pixel-level AUPRO"
            per_pixel_proauc1 = cal_pxl_pro(gt_mask, scores1)
            per_pixel_proauc2 = cal_pxl_pro(gt_mask, scores2)
            best_pxl_pro1 = per_pixel_proauc1 if per_pixel_proauc1 > best_pxl_pro1 else best_pxl_pro1
            best_pxl_pro2 = per_pixel_proauc2 if per_pixel_proauc2 > best_pxl_pro2 else best_pxl_pro2

            print(f"img-roc1: {img_roc_auc1:.3f} | img-roc2: {img_roc_auc2:.3f}")
            print(f"pix-roc1: {per_pixel_rocauc1:.3f} | pix-roc2: {per_pixel_rocauc2:.3f}")
            print(f"pix-pro1: {per_pixel_proauc1:.3f} | pix-pro2: {per_pixel_proauc2:.3f}")

        # print("image ROCAUC: %.3f" % best_img_roc1)
        # print("pixel ROCAUC: %.3f" % best_pxl_roc1)
        # print("pixel ROCAUC: %.3f" % best_pxl_pro1)

        total_roc_auc.append(best_img_roc1)
        total_pixel_roc_auc.append(best_pxl_roc1)
        total_pixel_pro_auc.append(best_pxl_pro1)

        # fig_pixel_rocauc.plot(fpr1, tpr1, label="%s ROCAUC: %.3f" % (class_name, per_pixel_rocauc1))
        # save_dir = args.save_path + "/" + f"pictures_{args.cnn}"
        # os.makedirs(save_dir, exist_ok=True)
        # plot_fig(test_imgs, scores1, gt_mask_list, threshold1, save_dir, class_name)

    # print("Average ROCAUC: %.3f" % np.mean(total_roc_auc))
    # fig_img_rocauc.title.set_text("Average image ROCAUC: %.3f" % np.mean(total_roc_auc))
    # fig_img_rocauc.legend(loc="lower right")

    # print("Average pixel ROCUAC: %.3f" % np.mean(total_pixel_roc_auc))
    # fig_pixel_rocauc.title.set_text("Average pixel ROCAUC: %.3f" % np.mean(total_pixel_roc_auc))
    # fig_pixel_rocauc.legend(loc="lower right")

    # print("Average pixel PROUAC: %.3f" % np.mean(total_pixel_pro_auc))

    # fig.tight_layout()
    # fig.savefig(os.path.join(args.save_path, "roc_curve.png"), dpi=100)


if __name__ == "__main__":
    run()
