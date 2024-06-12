"""Run tiled ensemble training."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.pipelines.tiled_ensemble.pipeline import TrainTiledEnsemble

if __name__ == "__main__":
    TrainTiledEnsemble().run()
