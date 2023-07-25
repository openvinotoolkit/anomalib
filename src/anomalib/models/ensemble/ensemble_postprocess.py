"""Classes and functions used for various postprocessing of ensemble predictions."""
from abc import ABC
from typing import Any

import torch

from anomalib.models.ensemble.ensemble_prediction_joiner import EnsemblePredictionJoiner
from anomalib.utils.metrics import AnomalyScoreThreshold, MinMax


# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class EnsemblePostProcess(ABC):
    def __init__(self):
        pass

    def process(self):
        pass


class SmoothJoins(EnsemblePostProcess):
    pass


class Normalize(EnsemblePostProcess):
    pass


class Threshold(EnsemblePostProcess):
    pass


class EnsemblePostProcessPipeline:
    pass


class PostProcessStats:
    def __init__(self, prediction_joiner: EnsemblePredictionJoiner):
        self.image_threshold = AnomalyScoreThreshold().cpu()
        self.pixel_threshold = AnomalyScoreThreshold().cpu()
        self.pixel_update_called = False

        self.minmax = MinMax().cpu()
        self.joiner = prediction_joiner

    def _update(self, output: Any):
        if "anomaly_maps" in output:
            self.minmax(output["anomaly_maps"])
        elif "box_scores" in output:
            self.minmax(torch.cat(output["box_scores"]))
        elif "pred_scores" in output:
            self.minmax(output["pred_scores"])
        else:
            raise ValueError("No values found for normalization, provide anomaly maps, bbox scores, or image scores")

        self.image_threshold.update(output["pred_scores"], output["label"].int())
        if "mask" in output.keys() and "anomaly_maps" in output.keys():
            self.pixel_threshold.update(torch.squeeze(output["anomaly_maps"]), torch.squeeze(output["mask"].int()))
            self.pixel_update_called = True

    def compute(self) -> None:

        for batch_index in range(self.joiner.ensemble_predictions.num_batches):
            joined_batch = self.joiner.join_tile_predictions(batch_index=batch_index)
            self._update(joined_batch)

        self.image_threshold.compute()
        if self.pixel_update_called:
            self.pixel_threshold.compute()
        else:
            self.pixel_threshold.value = self.image_threshold.value
