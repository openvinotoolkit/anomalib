STFPM
-------

This is the implementation of the `STFPM <https://arxiv.org/pdf/2103.04257.pdf>`_ paper.

Model Type: Segmentation

Description
***********

STFPM algorithm which consists of a pre-trained teacher network and a student network with identical architecture. The student network learns the distribution of anomaly-free images by matching the features with the counterpart features in the teacher network. Multi-scale feature matching is used to enhance robustness. This hierarchical feature matching enables the student network to receive a mixture of multi-level knowledge from the feature pyramid thus allowing for anomaly detection of various sizes.

During inference, the feature pyramids of teacher and student networks are compared. Larger difference indicates a higher probability of anomaly occurrence.

Architecture
************

.. image:: ../../images/stfpm/architecture.jpg
    :alt: STFPM Architecture

Usage
*****

.. code-block:: bash

    $ python tools/train.py --model stfpm


.. automodule:: anomalib.models.stfpm.torch_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: anomalib.models.stfpm.lightning_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: anomalib.models.stfpm.anomaly_map
   :members:
   :undoc-members:
   :show-inheritance:
