from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import nn
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


class FlowExtractor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights)
        self.transforms = weights.transforms()

    def pre_process(self, first_frame, last_frame):
        """Apply transforms."""
        # TODO: keep aspect ratio when resizing
        first_frame = F.resize(first_frame, size=[520, 960], antialias=False)
        last_frame = F.resize(last_frame, size=[520, 960], antialias=False)
        return self.transforms(first_frame, last_frame)

    def forward(self, first_frame, last_frame):
        _batch_size, _channels, height, width = first_frame.shape

        # preprocess batch
        first_frame, last_frame = self.pre_process(first_frame, last_frame)

        # get flow maps
        with torch.no_grad():
            flows = self.model(first_frame, last_frame)[-1]

        # convert back to original size
        flows = F.resize(flows, [height, width])

        return flows

    @staticmethod
    def plot(imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()
