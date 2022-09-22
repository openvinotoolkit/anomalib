CFlow
=====

This is the implementation of the `CFlow <https://arxiv.org/pdf/2107.12571v1.pdf>`_ paper.

Model Type: Segmentation

Description
***********

CFLOW model is based on a conditional normalizing flow framework adopted for anomaly detection with localization. It consists of a discriminatively pretrained encoder followed by a multi-scale generative decoders. The encoder extracts features with multi-scale pyramid pooling to capture both global and local semantic information with the growing from top to bottom receptive fields. Pooled features are processed by a set of decoders to explicitly estimate likelihood of the encoded features. The estimated multi-scale likelyhoods are upsampled to input size and added up to produce the anomaly map.

Architecture
************

.. image:: ../../images/cflow/architecture.jpg
    :alt: CFlow Architecture

Usage
*****

.. code-block:: bash

    python tools/train.py --model cflow

.. automodule:: anomalib.models.cflow.torch_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: anomalib.models.cflow.lightning_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: anomalib.models.cflow.anomaly_map
   :members:
   :undoc-members:
   :show-inheritance:
