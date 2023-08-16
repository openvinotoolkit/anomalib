import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

##
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

##
import matplotlib.ticker as mtick


def plot_sample_cv2(names, imgs, scores_: dict, gts, save_folder=None):
    # get subplot number
    total_number = len(imgs)

    scores = scores_.copy()
    # normarlisze anomalies
    for k, v in scores.items():
        max_value = np.max(v)
        min_value = np.min(v)

        scores[k] = (scores[k] - min_value) / max_value * 255
        scores[k] = scores[k].astype(np.uint8)
    # draw gts
    mask_imgs = []
    for idx in range(total_number):
        gts_ = gts[idx]
        mask_imgs_ = imgs[idx].copy()
        mask_imgs_[gts_ > 0.5] = (0, 0, 255)
        mask_imgs.append(mask_imgs_)

    # save imgs
    for idx in range(total_number):
        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_ori.jpg'), imgs[idx])
        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_gt.jpg'), mask_imgs[idx])

        for key in scores:
            heat_map = cv2.applyColorMap(scores[key][idx], cv2.COLORMAP_JET)
            visz_map = cv2.addWeighted(heat_map, 0.5, imgs[idx], 0.5, 0)
            cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_{key}.jpg'),
                        visz_map)


def plot_anomaly_score_distributions(scores: dict, ground_truths_list, save_folder, class_name):
    ground_truths = np.stack(ground_truths_list, axis=0)

    N_COUNT = 100000

    for k, v in scores.items():
        layer_score = np.stack(v, axis=0)
        normal_score = layer_score[ground_truths == 0]
        abnormal_score = layer_score[ground_truths != 0]

        plt.clf()
        plt.figure(figsize=(4, 3))
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        # with plt.style.context(['science', 'ieee', 'no-latex']):
        sns.histplot(np.random.choice(normal_score, N_COUNT), color="green", bins=50, label='${d(p_n)}$',
                     stat='probability', alpha=.75)
        sns.histplot(np.random.choice(abnormal_score, N_COUNT), color="red", bins=50, label='${d(p_a)}$',
                     stat='probability', alpha=.75)

        plt.xlim([0, 3])

        save_path = os.path.join(save_folder, f'distributions_{class_name}_{k}.jpg')

        plt.savefig(save_path, bbox_inches='tight', dpi=300)


valid_feature_visualization_methods = ['TSNE', 'PCA']


def visualize_feature(features, labels, legends, n_components=3, method='TSNE'):
    assert method in valid_feature_visualization_methods
    assert n_components in [2, 3]

    if method == 'TSNE':
        model = TSNE(n_components=n_components)
    elif method == 'PCA':
        model = PCA(n_components=n_components)

    else:
        raise NotImplementedError

    feat_proj = model.fit_transform(features)

    if n_components == 2:
        ax = scatter_2d(feat_proj, labels)
    elif n_components == 3:
        ax = scatter_3d(feat_proj, labels)
    else:
        raise NotImplementedError

    plt.legend(legends)
    plt.axis('off')


def scatter_3d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes(projection='3d')

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter3D(feat_proj[label == l, 0],
                      feat_proj[label == l, 1],
                      feat_proj[label == l, 2], s=5)

    return ax1


def scatter_2d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes()

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter(feat_proj[label == l, 0],
                    feat_proj[label == l, 1], s=5)

    return ax1
