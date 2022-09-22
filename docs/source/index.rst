
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

   tutorials/index
   how_to_guides/index
   reference_guide/index
   developer_guide/index

Getting Started
---------------

The Getting Started section contains all the necessary information regarding
setting up and installing the package. It also includes steps needed to train
the models.

Models
------

This page contains all the models implemented in the repository as well as
their API. It is the developer's responsibility to update this page when a new
model is added to the repo.

API Reference
-------------

This page lists all the modules, classes and functions available within the
`anomalib` package. This page is update automatically for the following modules:

- config
- core
- datasets
- hpo
- loggers
- utils

If a change is made to any of these modules, then the document will be
automatically updated. However, if a new module is introduced, then it must be
added to `api_references.rst`.

Tutorials
---------

This contains specific examples of using a feature or a model in the library.
These are written from the perspective of beginners.

Guides
------

This section contains all the guides written from the perspective of a new user
or if someone is trying to get some clarity on a topic.

Research
--------

Contains all the resources related to research such as benchmarks and papers.
These are written from the perspective of researchers.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
