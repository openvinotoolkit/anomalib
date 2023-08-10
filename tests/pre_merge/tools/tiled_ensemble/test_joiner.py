"""Test working of tile joiner"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.data import get_datamodule


class TestBasicJoiner:
    def test_tile_joining(self, get_ens_config, get_joiner, get_ensemble_predictions):
        config = get_ens_config
        joiner = get_joiner
        predictions = get_ensemble_predictions

        # prepared original data
        datamodule = get_datamodule(config)
        datamodule.prepare_data()
        datamodule.setup()
        original_data = next(iter(datamodule.test_dataloader()))

        batch = predictions.get_batch_tiles(0)

        joined_image = joiner.join_tiles(batch, "image")
        assert joined_image.equal(original_data["image"])

        joined_mask = joiner.join_tiles(batch, "mask")
        assert joined_mask.equal(original_data["mask"])

    def test_label_and_score_joining(self, get_joiner):
        joiner = get_joiner
        scores = torch.rand(4, 10)
        labels = scores > 0.5

        mock_data = {(0, 0): {}, (0, 1): {}, (1, 0): {}, (1, 1): {}}

        for i, data in enumerate(mock_data.values()):
            data["pred_scores"] = scores[i]
            data["pred_labels"] = labels[i]

        joined = joiner.join_labels_and_scores(mock_data)

        assert joined["pred_scores"].equal(scores.mean(dim=0))

        assert joined["pred_labels"].equal(labels.any(dim=0))

    def test_box_joining(self, get_joiner):
        joiner = get_joiner

        mock_data = {
            (0, 0): {
                "pred_boxes": [torch.ones(2, 4), torch.zeros(0, 4)],
                "box_scores": [torch.ones(2), torch.tensor([])],
                "box_labels": [torch.ones(2).type(torch.bool), torch.tensor([])],
            },
            (0, 1): {
                "pred_boxes": [torch.ones(1, 4), torch.ones(1, 4)],
                "box_scores": [torch.ones(1), torch.ones(1)],
                "box_labels": [torch.ones(1).type(torch.bool), torch.ones(1).type(torch.bool)],
            },
        }

        joined = joiner.join_boxes(mock_data)

        assert joined["pred_boxes"][0].shape == (3, 4)
        assert joined["box_scores"][0].shape == (3,)
        assert joined["box_labels"][0].shape == (3,)

        assert joined["pred_boxes"][1].shape == (1, 4)
        assert joined["box_scores"][1].shape == (1,)
        assert joined["box_labels"][1].shape == (1,)
