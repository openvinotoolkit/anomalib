"""F1 Score metric.

This is added for convenience.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from torchmetrics.classification import BinaryF1Score


class F1Score(BinaryF1Score):
    """This is a wrapper around torchmetrics' BinaryF1Score.

    The idea behind this is to retain the current configuration otherwise the one from
    torchmetrics requires ``task`` as a parameter.
    """
