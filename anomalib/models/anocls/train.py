#  COSMONiO © All rights reserved.
#  This file is subject to the terms and conditions defined in file 'LICENSE.txt',
#  which is part of this source code package.

from typing import Optional, Tuple

import torch
from anocls.model import AnomalyClassificationModel
from anomalib.utils.time_tracker import TimeTracker
from attrdict import AttrDict
from torch.utils.data import DataLoader


class AnomalyClassificationTrainer:
    def __init__(self, model: AnomalyClassificationModel):
        self.train_tracker: Optional[TimeTracker] = None
        self.cancel_training = False
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model = model

    def fit(
        self,
        hparams: AttrDict,
        data_loader: DataLoader
    ) -> bool:
        """
        Creates a new normality model and trains the model on the provided dataset.

        :return: the trained KDEClassifier model, or previous model if training failed
        """
        self.cancel_training = False
        self.model.reset_training_model(hparams)

        self.train_tracker = TimeTracker(steps=len(data_loader))
        self.train_tracker.tick()

        # feature extraction
        for item in data_loader:

            if self.cancel_training:
                break
            # extract feature vector using pre-trained CNN
            feature_vector = self.model.feature_extractor(item.cuda()).detach().cpu()
            # stage extracted feature vector
            self.model.training_model.stage_features(feature_vector)
            self.train_tracker.tick()

        # Check the model if it is successful, otherwise return previous model
        improved = False
        if (
            hasattr(self.model.training_model, "feature_list")
            and isinstance(self.model.training_model.feature_list, list)
            and len(self.model.training_model.feature_list) > self.model.training_model.n_comps
            and not self.cancel_training
        ):
            improved = self.model.training_model.commit()
            # self.model = model

        self.train_tracker = None

        return improved

    def request_cancel_training(self):
        self.cancel_training = True
