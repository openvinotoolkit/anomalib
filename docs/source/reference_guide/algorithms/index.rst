.. _available models:

Algorithms
==========

.. toctree::
   :maxdepth: 1
   :caption: List of Models:

   cflow
   dfkde
   dfm
   draem
   fastflow
   ganomaly
   padim
   patchcore
   reverse_distillation
   stfpm


Feature extraction & (pre-trained) backbones
---------------------------------------------

Several models will use a pre-trained model to extract feature maps from its internal submodules -- the *backbone*.

All the pre-trained backbones come from the package  `PyTorch Image Models (timm) <https://github.com/rwightman/pytorch-image-models>`_ and are wrapped by the class FeatureExtractor.

For an introduction to timm, please check the `Getting Started with PyTorch Image Models (timm): A Practitioner’s Guide <https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055>`_, in particular the introduction about models and the section about `feature extraction <https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#b83b:~:text=ready%20to%20train!-,Feature%20Extraction,-timm%20models%20also>`_.

More information at the `section "Multi-scale Feature Maps (Feature Pyramid)" in timm's docummentation about feature extraction <https://rwightman.github.io/pytorch-image-models/feature_extraction/#multi-scale-feature-maps-feature-pyramid>`_.

.. tip::

   * Papers With Code has an interface to easily browse models available in timm: `https://paperswithcode.com/lib/timm <https://paperswithcode.com/lib/timm>`_

   * You can also find them with the python package function timm.list_models("resnet*", pretrained=True)

The backbone can be set in the config file, two examples below.

.. warning::

   Anomalib < v.0.4.0

.. code-block:: yaml

    model:
      name: cflow
      backbone: wide_resnet50_2
      pre_trained: true

.. warning::

   Anomalib > v.0.4.0 Beta - Subject to Change

.. code-block:: yaml

    model:
      class_path: anomalib.models.Cflow
      init_args:
        backbone: wide_resnet50_2
        pre_trained: true
