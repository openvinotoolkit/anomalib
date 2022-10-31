#  COSMONiO © All rights reserved.
#  This file is subject to the terms and conditions defined in file 'LICENSE.txt',
#  which is part of this source code package.

import torch


class Normalizer:
    def __init__(self):
        self.mean_cpu = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.std_cpu = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        if torch.cuda.is_available():
            self.mean_cuda = self.mean_cpu.cuda()
            self.std_cuda = self.std_cpu.cuda()

    def normalize(self, patches):
        assert torch.is_tensor(patches)
        assert patches.dim() == 4
        assert patches.shape[1] == 3
        assert patches.max() <= 1.0
        assert patches.min() >= 0.0

        if patches.is_cuda:
            patches = (patches - self.mean_cuda) / self.std_cuda
        else:
            patches = (patches - self.mean_cpu) / self.std_cpu

        return patches

    def unnormalize(self, patches):
        assert torch.is_tensor(patches)
        assert patches.dim() == 4
        assert patches.max() <= 1.0
        assert patches.min() >= -1.0

        if patches.is_cuda:
            patches = patches * self.std_cuda + self.mean_cuda
        else:
            patches = patches * self.std_cpu + self.mean_cpu

        assert patches.min() >= 0.0
        assert patches.max() <= 1.0

        return patches
