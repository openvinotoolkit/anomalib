Hyperparameter Optimization using SigOpt
========================================

This tutorial focuses on hyperparameter optimization (HPO) using `SigOpt <https://sigopt.com/>`_. For this example, the :ref:`models:STFPM` model will be used but you can refer to the :ref:`available models` page for a list of all the available models in this repository.

Understanding the ``hpo.py`` file
----------------------------------

The ``hpo.py`` file is the entry point for searching optimal hyperparameters. It takes in three parameters ``--model <model-name>``, ``--model_config_path <path to config.yaml>``, and ``--hpo_type <sweep, bayesian, evolutionary, etc>``. As of now, only ``sweep`` method is supported by ``hpo.py``. The ``--model_config_path`` option is optional and if left empty, will take the default configuration file defined by the model.

Understanding the `config.yaml` file
-------------------------------------

For the purpose of this tutorial, we are going to look at the configuration file located at ``anomalib/models/stfpm/config.yaml``. You can also use your own config file but you might need to set the other parameters correctly.

The HPO configuration of the file is produced below:

.. code-block:: yaml

    hyperparameter_search:
        observation_budget: 10
        project: anomaly
        parallel_workers: 1

        metric:
            name: Pixel-Level AUC
            objective: maximize

        parameters:
            lr:
                type: double
                min: 1e-3
                max: 1.0
            momentum:
                type: double
                min: 0
                max: 1
            patience:
                type: int
                min: 1
                max: 10
            weight_decay:
                type: double
                min: 1e-5
                max: 1e-3


To search for hyperparameters, SigOpt first needs to know a how many total combinations it should try. These are referred to as _observations_ SigOpt. In the configuration above, we are telling SigOpt to run a total of 10 times by setting ``observation_budget: 10``. Feel free to increase this value if you want. Next, we define the project in which this sweep will be recorded. Here, we tell the script to upload to ``anomaly`` project.

.. warning:: Make sure that you have a project with the same name as the one defined in the configuration file. If you do not do this, then you will get an error.

We are going to skip the ``parallel_workers`` part as it has not been implemented yet. But in the future, you will be able to run the observations in parallel.

The ``metric`` field defines the metric you want to optimize for. SigOpt allows you option between ``maximize`` and ``minimize``. In our case, we want to monitor the ``Pixel-Level AUC`` of the :ref:`models:STFPM` model and we want to maximize this metric.

.. warning:: This is only for the developers. Make sure that when you log your metrics, you set ``prog_bar=True`` in ``self.log()``. Otherwise ``trainer.test()`` returns an empty dict.

Now, comes the actual parameters for which we want to find the optimal values. The SigOpt can only try values which are accepted by the ``model`` during its creation. Here is the ``model`` section of the :ref:`models:STFPM`'s ``config.yaml``.

.. code-block:: yaml

    model:
        name: stfpm
        backbone: resnet18
        layers:
            - layer1
            - layer2
            - layer3
        lr: 0.4
        momentum: 0.9
        patience: 5
        weight_decay: 0.0001
        metric: Pixel-Level AUC

We can see that the model takes in values for learning rate, momentum, patience and weight decay. So let's try to optimize these. In the config file, we define a ``parameter`` key and define the data type and range for each of the parameters.

.. note:: Make sure that you name your parameters with the same exact name as defined in the ``model`` section of the ``config.yaml``.

SigOpt supports either ``double``, ``int`` or ``str``. Hence we define learning rate, momentum, and weight decay, as ``double``. Since patience can only take integer values, it is defined as ``int``. Then, we give the range between which we want SigOpt to suggest values. For example, to optimize the learning rate, we define the entry as:

.. code-block:: yaml

    parameters:
        lr:
            type: double
            min: 1e-3
            max: 1.0

.. warning:: When defining a type as ``double``, make sure that when using whole numbers, add a decimal so that the number is parsed as floating point. Eg: 0.0 instead of 0.

Finding Optimal Parameters
--------------------------

Now, all that's left is to run the optimizer. Since we are going to use the default configuration provided with stfpm, you can just use the model name. However, for completeness, here is the entire command.

``python hpo.py --model stfpm --model_config_path anomalib/models/stfpm/config.yaml --hpo_type sweep``

It should print a link to the SigOpt dashboard where you will be able to see the results of optimization.

Congratulations! You have made it to the end |:tada:|
