DFM
---

This is the implementation of `DFM <https://arxiv.org/pdf/1909.11786.pdf>`_ paper.

Model Type: Classification

Description
***********

Fast anomaly classification algorithm that consists of a deep feature extraction stage followed by anomaly classification stage consisting of PCA and class-conditional Gaussian Density Estimation.

Feature Extraction
##################

Features are extracted by feeding the images through a ResNet18 backbone, which was pre-trained on ImageNet. The output of the penultimate layer (average pooling layer) of the network is used to obtain a semantic feature vector with a fixed length of 2048.

Anomaly Detection
#################

In the anomaly classification stage, class-conditional PCA transformations and Gaussian Density models are learned. Two types of scores are calculated (i) Feature-reconstruction scores (norm of the difference between the high-dimensional pre-image of a reduced dimension feature and the original high-dimensional feature), and (ii) Negative log-likelihood under the learnt density models. Either of these scores can be used for anomaly detection.

Usage
*****

.. code-block:: bash

    $ python tools/train.py --model dfm


.. automodule:: anomalib.models.dfm.torch_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: anomalib.models.dfm.lightning_model
   :members:
   :undoc-members:
   :show-inheritance:
