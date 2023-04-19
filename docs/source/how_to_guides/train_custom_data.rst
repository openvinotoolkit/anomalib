Training with Custom Data
=========================

Anomalib supports a number of datasets in various formats, including the state-of-the-art anomaly detection benchmarks such as MVTec AD and BeanTech. For those who would like to use the library on their custom datasets, anomalib also provides a ``Folder`` datamodule that can load datasets from a folder on a file system. The scope of this post will be to train anomalib models on custom datasets using the ``Folder`` datamodule.

Step 1: Install Anomalib
------------------------

Option - 1 : PyPI
^^^^^^^^^^^^^^^^^
Anomalib can be installed from PyPI via the following:

.. code-block:: bash

    pip install anomalib


Option - 2: Editable Install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Alternatively, it is also possible to do editable install:

.. code-block:: bash
    git clone https://github.com/openvinotoolkit/anomalib.git
    cd anomalib
    pip install -e .


.. _collect-your-custom-dataset:

Step 2: Collect Your Custom Dataset
-----------------------------------
Anomalib supports multiple image extensions such as ``".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", and ".webp"``. A dataset can be collected from images that have any of these extensions.


Step 3: Format your dataset
---------------------------
Depending on the use-case and collection, custom datasets can have different formats, some of which are listed below:

* A dataset with good and bad images.
* A dataset with good and bad images as well as mask ground-truths for pixel-wise evaluation.
* A dataset with good and bad images that is already split into training and testing sets.

Each of these use-cases is addressed by anomalib's ``Folder`` datamodule. Let's focus on the first use-case as an example of end-to-end model training and inference. In this post, we will use the :ref:`hazelnut-toy-dataset` dataset which you can download from `here <https://openvinotoolkit.github.io/anomalib/_downloads/3f2af1d7748194b18c2177a34c03a2c4/hazelnut_toy.zip>`_. The dataset consists of several folders, each containing a set of images. The ``colour`` and the ``crack`` folders represent two kinds of defects. We can ignore the ``masks`` folder for now.

Load your data to the following directory structure. Anomalib will use all images in the ``colour`` folder as part of the validation dataset and then randomly split the good images for training and validation.

.. code-block:: bash
    Hazelnut_toy
    ├── colour
    └── good


Step 4: Modify Config File
--------------------------
A YAML configuration file is necessary to run training for Anomalib. The training configuration parameters are categorized into 5 sections: ``dataset``, ``model``, ``project``, ``logging``, ``trainer``.

To get Anomalib functionally working with a custom dataset, one only needs to change the ``dataset`` section of the configuration file.

Below is an example of what the dataset parameters would look like for our ``hazelnut_toy`` folder specified in :ref:`Step 2 <collect-your-custom-dataset>`.

Let's choose `Padim algorithm <https://arxiv.org/pdf/2011.08785.pdf>`_, copy the sample config and modify the dataset section.

.. code-block:: bash

    $ cp anomalib/models/padim/config.yaml custom_padim.yaml


.. code-block:: yaml

    # Replace the dataset configs with the following.
    dataset:
        name: hazelnut
        format: folder
        root: ./datasets/hazelnut_toy
        normal_dir: good # name of the folder containing normal images.
        abnormal_dir: colour # name of the folder containing abnormal images.
        task: classification # classification or segmentation
        mask_dir: null #optional
        normal_test_dir: null # optional
        extensions: null
        split_ratio: 0.2  # normal images ratio to create a test split
        seed: 0
        image_size: 256
        train_batch_size: 32
        eval_batch_size: 32
        num_workers: 8
        normalization: imagenet # data distribution to which the images will be normalized
        test_split_mode: from_dir # options [from_dir, synthetic]
        val_split_ratio: 0.5 # fraction of train/test images held out for validation (usage depends on val_split_mode)
        transform_config:
            train: null
            eval: null
        val_split_mode: from_test # determines how the validation set is created, options [same_as_test, from_test]
        tiling:
            apply: false
            tile_size: null
            stride: null
            remove_border_count: 0
            use_random_tiling: False
            random_tile_count: 16

        model:
            name: padim
            backbone: resnet18
            layer:
            - layer1
        ...


Step 5: Run Training
--------------------

As per the config file, move ``Hazelnut_toy`` to the datasets section in the main root directory of anomalib, and then run

.. code-block:: bash

    $ python tools/train.py --config custom_padim.yaml


Step 6: Interpret Results
-------------------------

Anomalib will print out results of the trained model on the validation dataset. The printed metrics are dependent on the task mode chosen. The classification example provided in this tutorial prints out two scores: F1 and AUROC. The F1 score is a metric which values both the precision and recall, more information on its calculation can be found in this `blog <https://towardsdatascience.com/understanding-accuracy-recall-precision-f1-scores-and-confusion-matrices-561e0f5e328c>`_.

.. note::

    Not only does Anomalib classify whether a part is defected or not, it can also be used to segment the defects as well. To do this, simply add a folder called ``mask`` at the same directory level as the ``good`` and ``colour`` folders. This folder should contain binary images for the defects in the ``colour`` folder. Here, the white pixels represent the location of the defect. Populate the mask field in the config file with ``mask`` and change the task to segmentation to see Anomalib segment defects.

.. code-block:: bash

    Hazelnut_toy
    ├── colour
    │  ├── 00.jpg
    │  ├── 01.jpg
    │  ...
    ├── good
    │  ├── 00.jpg
    │  ├── 01.jpg
    └── mask
    ├── 00.jpg
    ├── 01.jpg
    ...

Here is an example of the generated results for a toy dataset containing Hazelnut with colour defects.

.. image:: ../images/how_to_guides/train_custom_data/hazelnut_results.gif
    :align: center


Logging and Experiment Management
---------------------------------

While it is delightful to know how good your model performed on your preferred metric, it is even more exciting to see the predicted outputs. Anomalib provides a couple of ways to log and track experiments. These can be used individually or in a combination. As of the current release, you can save images to a local folder, or upload to comet, weights and biases, or TensorBoard.

To select where you would like to save the images, change the ``log_images`` parameter in the ``Visualization`` section in the config file to true.

For example, setting the following ``log_images: True`` will result in saving the images in the results folder as shown in the tree structure below:

.. code-block:: bash

    results
    └── padim
        └── Hazelnut_toy
            ├── images
            │   ├── colour
            │   │   ├── 00.jpg
            │   │   ├── 01.jpg
            │   │   └── ...
            │   └── good
            │       ├── 00.jpg
            │       ├── 01.jpg
            │       └── ...
            └── weights
                └── model.ckpt


Logging to Tensorboard and/or W&B
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To use TensorBoard and/or W&B logger and/or Comet logger, ensure that the logger parameter is set to ``comet``, ``tensorboard``, ``wandb`` or ``[tensorboard, wandb]`` in the ``logging`` section of the config file.

An example configuration for saving to TensorBoard is shown in the figure below. Similarly after setting logger to ``wandb`` or 'comet' you will see the images on your wandb and/or comet project dashboard.

.. code-block:: yaml

    visualization:
        show_images: False # show images on the screen
        save_images: False # save images to the file system
        log_images: True # log images to the available loggers (if any)
        image_save_path: null # path to which images will be saved
        mode: full # options: ["full", "simple"]

        logging:
        logger: [comet, tensorboard, wandb] #Choose any combination of these 3
        log_graph: false

.. image:: ../images/how_to_guides/train_custom_data/logging.gif
    :align: center


Hyper-Parameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is very rare to find a model which works out of the box for a particular dataset. However, fortunately, we support tools which work out of the box to help tune the models in Anomalib to your particular dataset. As of the publication of this blog post, Anomalib supports `weights and biases <https://wandb.ai/>`_ for hyperparameter optimization. To get started have a look at ``sweep.yaml`` located at ``tools/hpo``. It provides a sample of how one can define a hyperparameter sweep.

.. code-block:: yaml

    observation_budget: 10
    method: bayes
    metric:
    name: pixel_AUROC
    goal: maximize
    parameters:
    dataset:
        category: hazelnut
        image_size:
        values: [128, 256]
    model:
        backbone:
        values: [resnet18, wide_resnet50_2]

The observation_budget informs wandb about the number of experiments to run. The method section defines the kind of method to use for HPO search. For other available methods, have a look at `Weights and Biases <https://docs.wandb.ai/guides/sweeps/quickstart>`_ documentation. The parameters section contains dataset and model parameters. Any parameter defined here overrides the parameter in the original model configuration.

To run a sweep, you can just call,

.. code-block:: bash

    $ python tools/hpo/wandb_sweep.py   \
        --model padim                   \
        --config ./path_to_config.yaml  \
        --sweep_config tools/hpo/sweep.yaml

In case ``model_config`` is not provided, the script looks at the default config location for that model. Note, you will need to have logged into a wandb account to use HPO search and view the results.

A sample run is visible in the screenshot below.

.. image:: ../images/how_to_guides/train_custom_data/hpo.gif
    :align: center


Benchmarking
------------
To add to the suit of experiment tracking and optimization, anomalib also includes a benchmarking script for gathering results across different combinations of models, their parameters, and dataset categories. The model performance and throughputs are logged into a csv file that can also serve as a means to track model drift. Optionally, these same results can be logged to Weights and Biases and TensorBoard. A sample configuration file is shown in the screenshot below.

.. code-block:: yaml
    seed: 42
    compute_openvino: false
    hardware:
    - cpu
    - gpu
    writer:
    - wandb
    - tensorboard
    grid_search:
    dataset:
        category:
        - colour
        - crack
        image_size: [128, 256]
    model_name:
        - padim
        - stfpm

This configuration computes the throughput and performance metrics for CPU and GPU for two categories of the toy dataset for Padim and STFPM models. The dataset can be configured in the respective model configuration files. By default, ``compute_openvino`` is set to False to support instances where OpenVINO requirements are not installed in the environment. Once installed, this flag can be set to True to get throughput on OpenVINO optimized models. The writer parameter is optional and can be set to ``writer: []`` in case the user only requires a csv file without logging to TensorBoard or Weights and Biases. It is also a good practice to set a value of seed to ensure reproducibility across runs and thus, is set to a non-zero value by default.

Once a configuration is decided, benchmarking can easily be performed by calling

.. code-block:: bash

    python tools/benchmarking/benchmark.py \
        --config tools/benchmarking/benchmark_params.yaml

A nice feature about the provided benchmarking script is that if the host system has multiple GPUs, the runs are parallelized over all the available GPUs for faster collection of result.

.. attention::

    Intel researchers actively maintain the Anomalib repository. Their mission is to provide the AI community with best-in-class performance and accuracy while also providing a positive developer experience. Check out the repo and start using anomalib right away!
