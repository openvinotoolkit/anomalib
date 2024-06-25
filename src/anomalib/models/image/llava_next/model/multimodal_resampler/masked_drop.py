import torch
import torch.nn as nn

import random


class MaskedDrop(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.mode = model_args.mm_mask_drop_mode
        self.skip_percentage = model_args.mm_mask_drop_skip_percentage
        self.ratio = model_args.mm_mask_drop_ratio
        self.ratio_upper = model_args.mm_mask_drop_ratio_upper
        self.ratio_lower = model_args.mm_mask_drop_ratio_lower

    def forward(self, image_features, *args, **kwargs):

        if not self.training:
            return image_features

        if self.skip_percentage > random.random():
            return image_features

        masked_features = []

        for image_feature in image_features:
            num_tokens = image_feature.shape[0]
            if self.mode == "fixed":
                num_keep = int(num_tokens * self.ratio)
                masked_features.append(self.random_masking(image_feature.unsqueeze(0), num_keep)[0][0])
            elif self.mode == "range":
                num_keep = int(num_tokens * random.uniform(self.ratio_lower, self.ratio_upper))
                masked_features.append(self.random_masking(image_feature.unsqueeze(0), num_keep)[0])
            elif self.mode == "cls_only":
                masked_features.append(image_feature[0:1])
            else:
                raise ValueError(f"Unexpected masked drop mode: {self.mode}")

        if self.mode not in ["range"] and (type(image_features) is not list or self.mode in ["cls_only"]):
            masked_features = torch.stack(masked_features, dim=0)

        return masked_features

    @property
    def config(self):
        return {
            "mm_resampler_type": "masked_drop",
            "mm_mask_drop_mode": self.mode,
            "mm_mask_drop_skip_percentage": self.skip_percentage,
            "mm_mask_drop_ratio": self.ratio,
            "mm_mask_drop_ratio_upper": self.ratio_upper,
            "mm_mask_drop_ratio_lower": self.ratio_lower,
        }

    def random_masking(self, x, len_keep):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
