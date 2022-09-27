Export & Optimization
--------------
This page will explain how to export your trained models to ONNX and OpenVINO format, and how the performance of the exported OpenVINO models can be optimized. For an explanation how the exported models can be deployed, please refer to the inference guide: :ref:`_inference_documentation`.

Export
=======
Anomalib models are fully compatible with the OpenVINO framework for accelerating inference on intel hardware. To export a model to OpenVINO format, simply set the export mode to ``openvino`` in the model config as shown below, and trigger a training run. When the training finishes, the trained model weights will be converted to OpenVINO Intermediate Representation (IR) format, and written to the file system in the chosen results folder. Since the OpenVINO model optimizer uses the ONNX format in one of the conversion steps, the ONNX model will be written to the file system as well.

.. code-block:: none
    :caption: Add this configuration to your config.yaml file to export your model to OpenVINO IR after training.

    optimization:
      export_mode: openvino

As a prerequisite, make sure that all required packages listed in ``requirements/openvino.txt`` are installed in your environment.

It is also possible to only write the ONNX model to the filesystem. This is done by setting the ``export_mode`` parameter to ``onnx``:

.. code-block:: none
    :caption: Add this configuration to your config.yaml file to export your model to ONNX format after training.

    optimization:
      export_mode: onnx

Optimization
=============
Anomalib supports OpenVINO's Neural Network Compression Framework (NNCF) to further improve the performance of the exported OpenVINO models. NNCF optimizes the neural network components of the anomaly models during the training process, and can therefore achieve a better performance-accuracy trade-off than post-training approaches.

.. note::
    NNCF support is in experimental stage, and is currently only available for the STFPM model

To enable NNCF, add the following configuration to your ``config.yaml``:

.. code-block:: none
    :caption: Add this configuration to your config.yaml file to enable NNCF.

    optimization:
      nncf:
        apply: true

The compressed model will be stored in the OpenVINO IR format in the specified results directory.
