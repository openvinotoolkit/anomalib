import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import os


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
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

        fig = plt.figure()
        ax0 = fig.add_subplot(221)
        ax0.axis("off")
        ax0.imshow(img)
        ax0.title.set_text("Image")

        ax1 = fig.add_subplot(222)
        ax1.axis("off")
        ax1.imshow(gt, cmap="gray")
        ax1.title.set_text("GroundTruth")

        ax2 = fig.add_subplot(223)
        ax2.axis("off")
        ax2.imshow(img, cmap="gray", interpolation="none")
        ax2.imshow(heat_map, cmap="jet", alpha=0.5, interpolation="none")
        ax2.title.set_text("Predicted heat map")

        ax3 = fig.add_subplot(224)
        ax3.axis("off")
        ax3.imshow(vis_img)
        ax3.title.set_text("Segmentation result")

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, class_name + "_{}".format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.0).astype(np.uint8)

    return x
