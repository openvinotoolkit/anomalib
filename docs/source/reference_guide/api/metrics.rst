Metrics
=======

There are two ways of configuring metrics in the config file:

1. a list of metric names, or
2. a mapping of metric names to class path and init args.

Each subsection in the section ``metrics`` of the config file can have a different style but inside each one it must be the same style.

.. code-block:: yaml
   :caption: Example of metrics configuration section in the config file.

   metrics:
      # imagewise metrics using the list of metric names style
      image:
         - F1Score
         - AUROC
      # pixelwise metrics using the mapping style
      pixel:
         F1Score:
            class_path: torchmetrics.F1Score
            init_args:
            compute_on_cpu: true
         AUROC:
            class_path: anomalib.utils.metrics.AUROC
            init_args:
            compute_on_cpu: true

List of metric names
--------------------

A list of strings that match the name of a class in ``anomalib.utils.metrics`` or ``torchmetrics`` (in this order of priority), which will be instantiated with default arguments.

Mapping of metric names to class path and init args
---------------------------------------------------

A mapping of metric names (str) to a dictionary with two keys: "class_path" and "init_args".

"class_path" is a string with the full path to a metric (from root package down to the class name, e.g.: "anomalib.utils.metrics.AUROC").

"init_args" is a dictionary of arguments to be passed to the class constructor.

.. automodule:: anomalib.utils.metrics
   :members:
   :undoc-members:
   :show-inheritance:
