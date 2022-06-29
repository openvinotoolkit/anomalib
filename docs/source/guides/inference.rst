.. _inference_documentation:

Inference
---------
Anomalib provides entrypoint scripts for using a trained model to generate predictions from a source of image data. This guide explains how to run inference with the standard PyTorch model and the exported OpenVINO model.


PyTorch (Lightning) Inference
==============
The entrypoint script in ``tools/inference/lightning.py`` can be used to run inference with a trained PyTorch model. The script runs inference by loading a previously trained model into a PyTorch Lightning trainer and running the ``predict sequence``. The entrypoint script has several command line arguments that can be used to configure inference:

+---------------------+----------+-------------------------------------------------------------------------------------+
| Parameter           | Required | Description                                                                         |
+=====================+==========+=====================================================================================+
| config              | True     | Path to the model config file.                                                      |
+---------------------+----------+-------------------------------------------------------------------------------------+
| weight_path         | True     | Path to the ``.ckpt`` model checkpoint file.                                        |
+---------------------+----------+-------------------------------------------------------------------------------------+
| image_path          | True     | Path to the image source. This can be a single image or a folder of images.         |
+---------------------+----------+-------------------------------------------------------------------------------------+
| save_path           | False    | Path to which the output images should be saved.                                    |
+---------------------+----------+-------------------------------------------------------------------------------------+
| visualization_mode  | False    | Determines how the inference results are visualized. Options: "full", "simple".     |
+---------------------+----------+-------------------------------------------------------------------------------------+
| disable_show_images | False    | When this flag is passed, visualizations will not be shown on the screen.           |
+---------------------+----------+-------------------------------------------------------------------------------------+

To run inference, call the script from the command line with the with the following parameters, e.g.:

``python tools/inference/lightning.py --config padim.yaml --weight_path results/weights/model.ckpt --image_path image.png``

This will run inference on the specified image file or all images in the folder. A visualization of the inference results including the predicted heatmap and segmentation results (if applicable), will be displayed on the screen, like the example below.



OpenVINO Inference
==============
To run OpenVINO inference, first make sure that your model has been exported to the OpenVINO IR format. Once the model has been exported, OpenVINO inference can be triggered by running the OpenVINO entrypoint script in ``tools/inference/openvino.py``. The command line arguments are very similar to PyTorch inference entrypoint script:

+-------------+----------+-------------------------------------------------------------------------------------+
| Parameter   | Required | Description                                                                         |
+=============+==========+=====================================================================================+
| config      | True     | Path to the model config file.                                                      |
+-------------+----------+-------------------------------------------------------------------------------------+
| weight_path | True     | Path to the OpenVINO IR model file (either ``.xml`` or ``.bin``)                    |
+-------------+----------+-------------------------------------------------------------------------------------+
| image_path  | True     | Path to the image source. This can be a single image or a folder of images.         |
+-------------+----------+-------------------------------------------------------------------------------------+
| save_data   | False    | Path to which the output images should be saved. Leave empty for live visualization.|
+-------------+----------+-------------------------------------------------------------------------------------+
| meta_data   | True     | Path to the JSON file containing the model's meta data (e.g. normalization          |
|             |          | parameters and anomaly score threshold).                                            |
+-------------+----------+-------------------------------------------------------------------------------------+

For correct inference results, the ``meta_data`` argument should be specified and point to the ``meta_data.json`` file that was generated when exporting the OpenVINO IR model. The file is stored in the same folder as the ``.xml`` and ``.bin`` files of the model.

As an example, OpenVINO inference can be triggered by the following command:

``python tools/inference/openvino.py --config padim.yaml --weight_path results/openvino/model.xml --image_path image.png --meta_data results/openvino/meta_data.json``

Similar to PyTorch inference, the visualization results will be displayed on the screen, and optionally saved to the file system location specified by the ``save_data`` parameter.
