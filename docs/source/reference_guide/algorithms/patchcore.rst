PatchCore
----------

This is the implementation of the `PatchCore <https://arxiv.org/pdf/2106.08265.pdf>`_ paper.

Model Type: Segmentation

Description
***********

The PatchCore algorithm is based on the idea that an image can be classified as anomalous as soon as a single patch is anomalous. The input image is tiled. These tiles act as patches which are fed into the neural network. It consists of a single pre-trained network which is used to extract "mid" level features patches. The "mid" level here refers to the feature extraction layer of the neural network model. Lower level features are generally too broad and higher level features are specific to the dataset the model is trained on. The features extracted during training phase are stored in a memory bank of neighbourhood aware patch level features.

During inference this memory bank is coreset subsampled. Coreset subsampling generates a subset which best approximates the structure of the available set and allows for approximate solution finding. This subset helps reduce the search cost associated with nearest neighbour search. The anomaly score is taken as the maximum distance between the test patch in the test patch collection to each respective nearest neighbour.

Architecture
************

.. image:: ../../images/patchcore/architecture.jpg
    :alt: PatchCore Architecture

Usage
*****

.. code-block:: bash

    $ python tools/train.py --model patchcore


.. automodule:: anomalib.models.patchcore.torch_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: anomalib.models.patchcore.lightning_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: anomalib.models.patchcore.anomaly_map
   :members:
   :undoc-members:
   :show-inheritance:
