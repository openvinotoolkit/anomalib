"""F1 Score metric.

This is added for convenience.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Literal

from torchmetrics.classification import BinaryF1Score

logger = logging.getLogger(__name__)


class F1Score(BinaryF1Score):
    """This is a wrapper around torchmetrics' BinaryF1Score.

    The idea behind this is to retain the current configuration otherwise the one from
    torchmetrics requires ``task`` as a parameter.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        multidim_average: Literal["global"] | Literal["samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(threshold, multidim_average, ignore_index, validate_args, **kwargs)
        logger.warning(
            "F1Score class exists for backwards compatibility. It will be removed in v1.1."
            " Please use BinaryF1Score from torchmetrics instead",
        )
