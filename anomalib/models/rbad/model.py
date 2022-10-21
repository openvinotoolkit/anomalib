#  COSMONiO © All rights reserved.
#  This file is subject to the terms and conditions defined in file 'LICENSE.txt',
#  which is part of this source code package.

import copy
import sys
import time

import numpy as np
import torch
from anomalib.utils.timer import Timer
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def process_features_using_density_filter(norm_model, features):
    assert norm_model.feature_count_since_last_update <= norm_model.reference_model_update_interval

    max_feature_count = norm_model.reference_model_update_interval - norm_model.feature_count_since_last_update
    features, spilled_features = np.split(features, [max_feature_count])

    if norm_model.feature_count_since_last_update == norm_model.reference_model_update_interval:
        print("\nreference_model_update_interval reached. Committing reference models...\n")
        norm_model.commit()
        norm_model.feature_count_since_last_update = 0

    if norm_model.kde_model is None:
        assert norm_model.pca_model is None
        norm_model.feature_list.append(features)
        return features, spilled_features
    else:
        densities = norm_model.evaluate(features, as_density=True, ln=True)
        low_density_mask = densities < norm_model.reference_model_threshold
        low_density_count = np.sum(low_density_mask)
        if low_density_count > 0:
            features_to_commit = features[low_density_mask]
            norm_model.feature_list.append(features_to_commit)
            return features_to_commit, spilled_features
        else:
            return np.empty((0, features.shape[1]), dtype=features.dtype), spilled_features


class NormalityModel:
    def __init__(
        self, n_comps: int = 16, pre_processing: str = "scale", filter_type: str = "count", filter_count: int = 40000
    ):
        self.feature_list = []
        self.region_counter = 0
        self.cancel_training: bool = False

        # MODEL
        self.pca_model = None
        self.kde_model = None
        self.n_comps = n_comps
        self.pre_processing = pre_processing
        self.max_length = -1.0

        # FILTER
        self.filter_type = filter_type
        self.filter_count = filter_count
        if self.filter_count == 0:
            self.filter_type = "__DISABLED__"

    def stage_features(self, features):
        if features is None or features.size == 0:
            return 0

        assert isinstance(features, np.ndarray) or torch.is_tensor(features)
        assert len(features.shape) == 2

        if torch.is_tensor(features):
            features = features.numpy()

        self.region_counter += features.shape[0]

        if self.filter_type in ["__DISABLED__", "count"]:
            self.feature_list.append(features)
            return features.shape[0]

    def commit(self):
        if len(self.feature_list) < 1:
            print("Not enough features to commit. Not making a model.")
            return False

        feature_stack = np.vstack(self.feature_list)
        if feature_stack.shape[0] < 2:
            print("Not enough features to commit. Not making a model.")
            return False

        # COMMIT
        start_time = time.time()

        self.pca_model = PCA(n_components=self.n_comps)
        if self.filter_type == "count" and feature_stack.shape[0] > self.filter_count:
            keep_indices = np.random.randint(0, feature_stack.shape[0], (self.filter_count,))
            feature_stack = feature_stack[keep_indices]
            self.region_counter = feature_stack.shape[0]

        print("\nPerforming PCA on {} data points...".format(feature_stack.shape[0]))
        feature_stack = self.pca_model.fit_transform(feature_stack)

        if self.pre_processing == "norm":
            feature_stack = normalize(feature_stack, axis=1, norm="l2")
        elif self.pre_processing == "scale":
            for feature_vec in feature_stack:
                l = np.linalg.norm(feature_vec)
                if l > self.max_length:
                    self.max_length = l
            feature_stack /= self.max_length
        elif self.pre_processing == "none":
            pass
        else:
            raise ValueError("Invalid preprocessing mode: {}".format(self.pre_processing))
        print("That took {:.2} seconds".format(time.time() - start_time))

        print("\nPerforming KDE on {} data points...".format(feature_stack.shape[0]))
        start_time = time.time()
        feature_stack = feature_stack.transpose()
        self.kde_model = gaussian_kde(feature_stack, bw_method="scott")
        print("That took {:.2} seconds\n".format(time.time() - start_time))
        return True

    def evaluate(self, sem_feats, as_density=False, ln=False, timer=Timer()):
        if self.pca_model is None or self.kde_model is None:
            if as_density:
                return np.ones((sem_feats.shape[0],), dtype=np.float64) * sys.float_info.max
            else:
                return np.zeros((sem_feats.shape[0],), dtype=np.float64)

        if torch.is_tensor(sem_feats):
            sem_feats = sem_feats.numpy()

        timer.start()
        if sem_feats.shape[0] == 0:
            return np.empty((0,))
        elif sem_feats.shape[0] == 1:
            sem_feats.reshape(1, -1)

        tmp = self.pca_model.transform(sem_feats)

        if self.pre_processing == "norm":
            tmp = normalize(tmp, axis=1, norm="l2")
        elif self.pre_processing == "scale":
            tmp /= self.max_length
        else:
            assert self.pre_processing == "none"

        # Using scipy
        tmp = tmp.transpose()
        tmp = self.kde_model.evaluate(tmp)

        tmp += 1e-300
        timer.stop()

        if as_density:
            return np.log(tmp) if ln else tmp
        else:
            return np.log(1.0 / tmp) if ln else 1.0 / tmp

    def feature_count(self):
        return sum(e.shape[0] for e in self.feature_list)

    def region_count(self):
        return self.region_counter

    def clone(self):
        return copy.deepcopy(self)
