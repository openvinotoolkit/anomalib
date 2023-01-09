"""Region-based Anomaly Detection with Real Time Training and Analysis."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from .lightning_model import Rkde, RkdeLightning

__all__ = ["Rkde", "RkdeLightning"]
