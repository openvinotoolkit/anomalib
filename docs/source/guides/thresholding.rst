.. _thresholding:

Thresholding
============

Anomalib provides various methods to compute thresholds.

Adaptive Thresholding
---------------------

Adaptive thresholding  computes the optimal threshold by finding a threshold that maxamizes the F1 Score on the validation set.
To use adaptive thresholding, you need to set the following keys in the `config.yaml` file under `metrics`.

.. code-block:: yaml
    :caption: Using only string

    ...
    metrics:
          ...
          threshold: adaptive

You can always pass default threshold but it will be overwritten during validation step.

.. code-block:: yaml
    :caption: By passing arguments

    ...
    metrics:
          ...
          threshold:
            adaptive:
                default_value: 0.5

Manual Thresholding
-------------------

You can also manually pass thresholds.

.. code-block:: yaml
    :caption: Manual Thresholding

    ...
    metrics:
          ...
          threshold:
            manual:
                image_threshold: 0.5
                pixel_threshold: 0.5

.. note::
    When using manual thresholding, you need to pass the image or pixel threshold. Otherwise, it will throw an error.
