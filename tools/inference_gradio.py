"""Anomalib Gradio Script.

This script provide a gradio web interface
"""

from typing import Tuple, Union
import gradio as gr
import gradio.inputs
import gradio.outputs
import numpy as np

import warnings
from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path

from omegaconf import DictConfig, ListConfig

from anomalib.config import get_configurable_parameters
from anomalib.deploy.inferencers.base import Inferencer

from anomalib.post_processing import Visualizer, compute_mask, superimpose_anomaly_map
from skimage.segmentation import mark_boundaries


def infer(image: np.ndarray, inferencer: Inferencer, threshold: int = 50.) -> Tuple[np.ndarray, 
                                                                                    float,
                                                                                    np.ndarray,
                                                                                    np.ndarray,
                                                                                    np.ndarray]:
    # Perform inference for the given image.
    threshold = threshold/100
    anomaly_map, anomaly_score = inferencer.predict(image=image, superimpose=False)
    heat_map = superimpose_anomaly_map(anomaly_map, image)
    pred_mask = compute_mask(anomaly_map, threshold)
    vis_img = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")
    return anomaly_map, anomaly_score, heat_map, pred_mask, vis_img


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a model config file")
    parser.add_argument("--weight_path", type=Path, required=True, help="Path to a model weights")
    parser.add_argument("--meta_data", type=Path, required=False, help="Path to JSON file containing the metadata.")
    
    parser.add_argument("--threshold", type=float, required=False, default=0.75, help="Value to threshold anomaly scores into 0-1 range")
    
    parser.add_argument("--share", type=bool, required=False, default=False, help="Share Gradio `share_url`")
    
    args = parser.parse_args()
    
    return args


def get_inferencer(args: Union[DictConfig, ListConfig]) -> Inferencer:
    config = get_configurable_parameters(config_path=args.config)
    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
    # for the openvino models.
    extension = args.weight_path.suffix
    inferencer: Inferencer
    if extension in (".ckpt"):
        module = import_module("anomalib.deploy.inferencers.torch")
        TorchInferencer = getattr(module, "TorchInferencer")  # pylint: disable=invalid-name
        inferencer = TorchInferencer(config=config, model_source=args.weight_path, meta_data_path=args.meta_data)

    elif extension in (".onnx", ".bin", ".xml"):
        module = import_module("anomalib.deploy.inferencers.openvino")
        OpenVINOInferencer = getattr(module, "OpenVINOInferencer")  # pylint: disable=invalid-name
        inferencer = OpenVINOInferencer(config=config, path=args.weight_path, meta_data_path=args.meta_data)

    else:
        raise ValueError(
            f"Model extension is not supported. Torch Inferencer exptects a .ckpt file,"
            f"OpenVINO Inferencer expects either .onnx, .bin or .xml file. Got {extension}"
        )
    
    return inferencer

if __name__ == "__main__":
    args = get_args()
    
    inferencer = get_inferencer(args)    
    
    iface = gr.Interface(fn=lambda image, threshold: infer(image, inferencer, threshold), 
                         inputs=
                            [
                                gradio.inputs.Image(shape=None,
                                                    image_mode="RGB",
                                                    source="upload",
                                                    tool="editor",
                                                    type="numpy",
                                                    label="Image"),
                                gradio.inputs.Slider(default=50.,
                                                     label="threshold",
                                                     optional=False)
                            ], 
                         outputs=
                            [
                                gradio.outputs.Image(
                                                    type="numpy",
                                                    label="Anomaly Map"),
                                gradio.outputs.Textbox(type="number",
                                                       label="Anomaly Score"),
                                gradio.outputs.Image(
                                                    type="numpy",
                                                    label="Predicted Heat Map"),
                                gradio.outputs.Image(
                                                    type="numpy",
                                                    label="Predicted Mask"),
                                gradio.outputs.Image(
                                                    type="numpy",
                                                    label="Segmentation Result"),
                            ],
                         title=f"Anomalib",
                         description=f"",
                         allow_screenshot=True
                         )
    
    iface.launch()