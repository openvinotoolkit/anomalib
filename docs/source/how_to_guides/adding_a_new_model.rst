Adding a New Model to Anomalib
==============================
Anomalib aims to have the implementation of the state-of-the-art algorithms published in the literature. To integrate an anomaly detection model into anomalib, the following steps should be followed:

* Create a new sub-package
* Create an ``__init__.py`` file.
* Create a ``config.yaml`` file.
* Create a ``torch_model.py`` file.
* Create a ``lightning_model.py`` file.
* [OPTIONAL] Create a ``loss.py`` file.
* [OPTIONAL] Create an ``anomaly_map.py`` file.
* Create a ``README.md`` file.


Create a New Sub Package
--------------------------
This is a new directory to be created in ``anomalib/models`` to store the model-related files. The overall outline would be as follows:

.. code-block:: bash

    ./anomalib/models/<new-model>
    ├── __init__.py
    ├── config.yaml
    ├── torch_model.py
    ├── lightning_model.py
    ├── loss.py    # OPTIONAL
    ├── anomaly_map.py    # OPTIONAL
    └── README.md


Create a ``config.yaml`` file
-------------------------------

This file stores all the configurations, from data to optimization options. An exemplary yaml file is shown below. For a full configuration file, you could refer to one of the existing model implementations, such as `patchcore configuration. <https://github.com/openvinotoolkit/anomalib/blob/main/anomalib/models/patchcore/config.yaml>`_

.. code-block:: yaml

    dataset:
        name: mvtec #options: [mvtec, btech, folder]
        format: mvtec
        ...
    model:
        name: patchcore
        backbone: wide_resnet50_2
        ...
    metrics:
        image:
            - F1Score
        ...
    visualization:
        show_images: False # show images on the screen
        ...
    # PL Trainer Args. Don't add extra parameter here.
    trainer:
        accelerator: auto # <"cpu", "gpu", "tpu", "ipu", "hpu", "auto">
        ...



Create a ``torch_model.py`` Module
----------------------------------
This file contains the torch model implementation that inherits from ``torch.nn.Module``, defines the model architecture and performs a basic forward-pass. The advantage of storing the model in a separate ``torch_model.py`` file is that the model is de-coupled from the rest of the anomalib implementations and could be used outside the library as well. Basic implementation would look like as follows:

.. code-block:: python

    class NewModelModel(nn.Module):
        """New Model Module."""
        def __init__(self):
            pass
        def forward(self, x):
            pass


Create a ``lightning_model.py`` Module
--------------------------------------
This module contains the lightning model implementation that inherits from the ``AnomalModule``, which already has the ``anomalib`` related attributes and methods. The user does not need to worry about the boilerplate code and only needs to implement the training and validation logic of the algorithm.

.. code-block:: python

    class NewModel(AnomalyModule):
        """PL Lightning Module for the New Model."""
        def __init__(self):
            super().__init__()
            pass
        def training_step(self, batch):
            pass
        ...
        def validation_step(self, batch):
            pass

Create a ``loss.py`` File - [Optional]
--------------------------------------
This module's availability is dependent on the algorithm. If the algorithm requires a custom, complex loss function, this file may contain the subclass of the torch.nn.Module class implementation. This loss would subsequently be utilized by the lightning module.

.. code-block:: python

    class NewModelLoss(nn.Module):
        """NewModel Loss."""

        def forward(self) -> Tensor:
            """Calculate the NewModel loss."""
            pass

Create an ``anomaly_map.py`` File - [Optional]
---------------------------------------------
Similar to the loss.py module, the anomaly map.py module is optional depending on the capabilities of the algorithm. This module may be implemented if the algorithm supports segmentation so that the location of the anomaly can be predicted pixel-by-pixel.

.. code-block:: python

    class AnomalyMapGenerator(nn.Module):
        """Generate Anomaly Heatmap."""

        def __init__(self, input_size: Union[ListConfig, Tuple]):
            pass

        def forward(self, x: Tensor) -> Tensor:
            """Generate Anomaly Heatmap."""
            ...
            return anomaly_map


Create a ``README.md`` File
---------------------------
Once the implementation is done, this readme file would describe the model using the following structure.

.. code-block:: markdown

    # Name of the Model

    ## Description
    Brief description of the paper.

    ## Architecture
    A diagram showing the high-level overview.

    ## Usage
    python tools/train.py --model <newmodel>

    ## Benchmark
    Benchmark results on MVTec categories.

Add Model to the Tests
----------------------
It is essential that newly added models do not disrupt the workflow and that their components are continually inspected. In this regard, the model will be added to our list of tested models.

To test the model, you need to add the model name `here <https://github.com/openvinotoolkit/anomalib/blob/main/tests/pre_merge/models/test_model_premerge.py#L18>`_.

The list of models to test would then become,

.. code-block:: python

    @pytest.mark.parametrize(
        ["model_name", "nncf"],
        [
            ("cflow", False),
            ("dfkde", False),
            ...
            ("newmodel", False),
        ],
    )
    @TestDataset(num_train=20, num_test=10)

This would check if the training works for the model. It is also important to check whether the inference capabilities of the model works as well. To do so, the model is to be added `here <https://github.com/openvinotoolkit/anomalib/blob/main/tests/pre_merge/deploy/test_inferencer.py>`_.

.. code-block:: python

    class TestInferencers:
    @pytest.mark.parametrize(
        "model_name",
        [
            "cflow",
            "dfkde",
            ...
            "newmodel"
        ],
    )

Add Model to the Docs
---------------------
Final step would be to add the model to the docs. To do so, one would create a ``newmodel.rst`` file in ``docs/reference_guide/algorithms``, and include it in ``docs/reference_guide/algorithms/index.rst`` as follows:

.. code-block:: sphinx

    .. _available models:

    Algorithms
    ==========

    .. toctree::
    :maxdepth: 3
    :caption: Contents:

    cflow
    dfkde
    ...
    newmodel

That is all! Now, the model would function flawlessly with anomalib!
