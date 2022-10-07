DFKDE
-------

Model Type: Classification

Description
***********

Fast anomaly classification algorithm that consists of a deep feature extraction stage followed by anomaly classification stage consisting of PCA and Gaussian Kernel Density Estimation.

Feature Extraction
##################

Features are extracted by feeding the images through a ResNet50 backbone, which was pre-trained on ImageNet. The output of the penultimate layer (average pooling layer) of the network is used to obtain a semantic feature vector with a fixed length of 2048.

Anomaly Detection
#################

In the anomaly classification stage, the features are first reduced to the first 16 principal components. Gaussian Kernel Density is then used to obtain an estimate of the probability density of new examples, based on the collection of training features obtained during the training phase.

Usage
*****

.. code-block:: bash

    python tools/train.py --model dfkde


.. automodule:: anomalib.models.dfkde.torch_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: anomalib.models.dfkde.lightning_model
   :members:
   :undoc-members:
   :show-inheritance:
