.. _available models:

Models Reference
=====================


Available Models
-----------------

Segmentation
*************

- `Padim`_
- `PatchCore`_
- `STFPM`_ (Supports OpenVINO)

Classification
***************

- `DFKDE`_
- `DFM`_

Loading Models
---------------

.. automodule:: anomalib.models
    :members:

.. currentmodule:: anomalib.models

Base
-----

.. autosummary::
    :toctree: models
    :nosignatures:

    base.lightning_modules
    base.torch_modules

DFKDE
-------

Deep Feature Kernel Density Estimation
***************************************

Fast anomaly classification algorithm that consists of a deep feature extraction stage followed by anomaly classification stage consisting of PCA and Gaussian Kernel Density Estimation.

Feature Extraction
******************

Features are extracted by feeding the images through a ResNet50 backbone, which was pre-trained on ImageNet. The output of the penultimate layer (average pooling layer) of the network is used to obtain a semantic feature vector with a fixed length of 2048.

Anomaly Detection
*******************

In the anomaly classification stage, the features are first reduced to the first 16 principal components. Gaussian Kernel Density is then used to obtain an estimate of the probability density of new examples, based on the collection of training features obtained during the training phase.

.. autosummary::
    :toctree: models
    :nosignatures:

    dfkde.model
    dfkde.normality_model

DFM
---

Probabilistic Modeling of Deep Features
****************************************

Fast anomaly classification algorithm that consists of a deep feature extraction stage followed by anomaly classification stage consisting of PCA and class-conditional Gaussian Density Estimation.

Feature Extraction
*******************

Features are extracted by feeding the images through a ResNet18 backbone, which was pre-trained on ImageNet. The output of the penultimate layer (average pooling layer) of the network is used to obtain a semantic feature vector with a fixed length of 2048.

Anomaly Detection
******************

In the anomaly classification stage, class-conditional PCA transformations and Gaussian Density models are learned. Two types of scores are calculated
    (i) Feature-reconstruction scores (norm of the difference between the high-dimensional pre-image of a reduced dimension feature and the original high-dimensional feature), and

    (ii) Negative log-likelihood under the learnt density models. Either of these scores can be used for anomaly detection.

.. autosummary::
    :toctree: models
    :nosignatures:

    dfm.model
    dfm.pca_model

Padim
------

.. autosummary::
    :toctree: models
    :nosignatures:

    padim.model

PatchCore
----------

.. autosummary::
    :toctree: models
    :nosignatures:

    patchcore.sampling_methods.kcenter_greedy
    patchcore.sampling_methods.sampling_def
    patchcore.model

STFPM
-------

.. autosummary::
    :toctree: models
    :nosignatures:

    stfpm.model
