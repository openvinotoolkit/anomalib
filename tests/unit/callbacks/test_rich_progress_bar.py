# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit test for MaxStepsProgressCallback picklability."""

import pickle  # noqa: S403

from lightning.pytorch import Trainer

from anomalib.callbacks import MaxStepsProgressCallback


def test_max_steps_progress_callback_is_picklable() -> None:
    """Test that trainer callbacks with MaxStepsProgressCallback can be pickled."""
    trainer = Trainer(max_steps=100)
    trainer.fit_loop.max_epochs = -1

    callback = MaxStepsProgressCallback()
    callback.on_train_start(trainer, None)  # type: ignore[arg-type]

    # Should serialize without error
    pickled = pickle.dumps(trainer.callbacks)  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
    assert len(pickled) > 0
