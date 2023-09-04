
.. image:: ./images/logos/anomalib-text.png
   :scale: 50 %
   :align: center

Anomalib is a deep learning library that aims to collect state-of-the-art anomaly detection algorithms for benchmarking on both public and private datasets. Anomalib provides several ready-to-use implementations of anomaly detection algorithms described in the recent literature, as well as a set of tools that facilitate the development and implementation of custom models. The library has a strong focus on image-based anomaly detection, where the goal of the algorithm is to identify anomalous images, or anomalous pixel regions within images in a dataset. Anomalib is constantly updated with new algorithms and training/inference extensions, so keep checking!

.. image:: ./images/readme.png
   :alt: Sample Image

Structure of the Documentation
==============================

This documentation is divided into the following sections:

.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Contents

   getting_started/index
   tutorials/index
   how_to_guides/index
   reference_guide/index
   developer_guide/index

Tutorials
---------
The ``Tutorials`` section contains all the necessary information regarding
setting up and installing the package. It also includes steps needed to train, export, infer models, perform HPO, benchmarking and logging.

Reference Guide
---------------
Algorithms
^^^^^^^^^^
This page contains all the models implemented in the repository as well as
their API. It is the developer's responsibility to update this page when a new
model is added to the repo.

API Reference
^^^^^^^^^^^^^
This page lists all the modules, classes and functions available within the
`anomalib` package. This page is update automatically for the following modules:

- cli
- config
- data
- model
- post_processing
- metrics
- loggers
- hpo
- callbacks

If a change is made to any of these modules, then the document will be
automatically updated. However, if a new algorithm is introduced, then it must be added to :ref:`available models`.


Developer Guide
---------------
This section contains all the guidelines for those who would like to contribute to the development of anomalib and to know more about the developer tools.


Citing the repository
=====================
You can cite this repository as

.. code-block:: tex

   @INPROCEEDINGS{anomalib,
      author={Akcay, Samet and Ameln, Dick and Vaidya, Ashwin and Lakshmanan, Barath and Ahuja, Nilesh and Genc, Utku},
      booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
      title={Anomalib: A Deep Learning Library for Anomaly Detection},
      year={2022},
      volume={},
      number={},
      pages={1706-1710},
      doi={10.1109/ICIP46576.2022.9897283}
   }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
