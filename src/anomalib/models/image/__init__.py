# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib Image Models.

This module contains implementations of various deep learning models for image-based
anomaly detection.

Example:
    >>> from anomalib.models.image import Padim, Patchcore
    >>> from anomalib.data import MVTecAD  # doctest: +SKIP
    >>> from anomalib.engine import Engine  # doctest: +SKIP

    >>> # Initialize model and data
    >>> datamodule = MVTecAD()  # doctest: +SKIP
    >>> model = Padim()  # doctest: +SKIP
    >>> # Train using the Engine

    >>> engine = Engine()  # doctest: +SKIP
    >>> engine.fit(model=model, datamodule=datamodule)  # doctest: +SKIP

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)  # doctest: +SKIP

Available Models:
    - :class:`AnomalyVFM`: Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors
    - :class:`AnomalyDINO`: Boost Memorybank Models with DINOv2
    - :class:`Cfa`: Contrastive Feature Aggregation
    - :class:`Cflow`: Conditional Normalizing Flow
    - :class:`CFM`: Crossmodal Feature Mapping
    - :class:`Csflow`: Conditional Split Flow
    - :class:`Dfkde`: Deep Feature Kernel Density Estimation
    - :class:`Dfm`: Deep Feature Modeling
    - :class:`Dinomaly`: DinoV2-based Reconstruction Error Model
    - :class:`Draem`: Dual Reconstruction by Adversarial Masking
    - :class:`Dsr`: Deep Spatial Reconstruction
    - :class:`EfficientAd`: Efficient Anomaly Detection
    - :class:`Fastflow`: Fast Flow
    - :class:`FoundAD`: Foundation Encoder Manifold Projector Anomaly Detection
    - :class:`Fre`: Feature Reconstruction Error
    - :class:`Ganomaly`: Generative Adversarial Networks
    - :class:`GeneralAD`: Attending to Distorted Features
    - :class:`GRDNet`: GRD-Net
    - :class:`L2BT`: Learning to Be a Transformer to Pinpoint Anomalies
    - :class:`MHPatchcore`: Mahalanobis PatchCore
    - :class:`Padim`: Patch Distribution Modeling
    - :class:`Patchcore`: Patch Core
    - :class:`Patchflow`: Patch Flow
    - :class:`Rad`: Retrieval-based Anomaly Detection
    - :class:`ReverseDistillation`: Reverse Knowledge Distillation
    - :class:`Stfpm`: Student-Teacher Feature Pyramid Matching
    - :class:`SuperADD`: Supervised Anomaly Detection with Additive Feature Fusion
    - :class:`SuperSimpleNet`: SuperSimpleNet
    - :class:`Uflow`: Unsupervised Flow
    - :class:`UniNet`: Student-Teacher Contrastive Learning Model
    - :class:`VlmAd`: Vision Language Model Anomaly Detection
    - :class:`WinClip`: Zero-/Few-Shot CLIP-based Detection
"""

from .anomaly_dino import AnomalyDINO
from .anomalyvfm import AnomalyVFM
from .cfa import Cfa
from .cflow import Cflow
from .cfm import CFM
from .csflow import Csflow
from .dfkde import Dfkde
from .dfm import Dfm
from .dinomaly import Dinomaly
from .draem import Draem
from .dsr import Dsr
from .efficient_ad import EfficientAd
from .fastflow import Fastflow
from .foundad import FoundAD
from .fre import Fre
from .ganomaly import Ganomaly
from .general_ad import GeneralAD
from .glass import Glass
from .grdnet import GRDNet
from .inp_former import InpFormer
from .l2bt import L2BT
from .mh_patchcore import MHPatchcore
from .padim import Padim
from .patchcore import Patchcore
from .patchflow import Patchflow
from .rad import Rad
from .reverse_distillation import ReverseDistillation
from .stfpm import Stfpm
from .super_add import SuperADD
from .supersimplenet import Supersimplenet
from .uflow import Uflow
from .uninet import UniNet
from .vlm_ad import VlmAd
from .winclip import WinClip

__all__ = [
    "AnomalyVFM",
    "AnomalyDINO",
    "Cfa",
    "Cflow",
    "CFM",
    "Csflow",
    "Dfkde",
    "Dfm",
    "Dinomaly",
    "Draem",
    "Dsr",
    "EfficientAd",
    "Fastflow",
    "FoundAD",
    "Fre",
    "Ganomaly",
    "GeneralAD",
    "Glass",
    "GRDNet",
    "InpFormer",
    "L2BT",
    "MHPatchcore",
    "Padim",
    "Patchcore",
    "Patchflow",
    "Rad",
    "ReverseDistillation",
    "Stfpm",
    "Supersimplenet",
    "SuperADD",
    "Uflow",
    "UniNet",
    "VlmAd",
    "WinClip",
]
