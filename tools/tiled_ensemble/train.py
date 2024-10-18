"""Run tiled ensemble training."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.pipelines.tiled_ensemble import EvalTiledEnsemble, TrainTiledEnsemble

if __name__ == "__main__":
    print("Running tiled ensemble train pipeline")
    train_pipeline = TrainTiledEnsemble()
    # run training
    train_pipeline.run()

    print("Running tiled ensemble test pipeline.")
    # pass the root dir from train run to load checkpoints
    test_pipeline = EvalTiledEnsemble(train_pipeline.root_dir)
    test_pipeline.run()
