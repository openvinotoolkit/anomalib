"""Anomalib Gradio Script.

This script provide a gradio web interface
"""

from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import gradio.inputs
import gradio.outputs
import numpy as np
from skimage.segmentation import mark_boundaries

from anomalib.config import get_configurable_parameters
from anomalib.deploy import Inferencer
from anomalib.post_processing import compute_mask, superimpose_anomaly_map


def infer(
    image: np.ndarray, inferencer: Inferencer, threshold: float = 50.0
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Inference function, return anomaly map, score, heat map, prediction mask ans visualisation.

    Args:
        image (np.ndarray): image to compute
        inferencer (Inferencer): model inferencer
        threshold (float, optional): threshold between 0 and 100. Defaults to 50.0.

    Returns:
        Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        anomaly_map, anomaly_score, heat_map, pred_mask, vis_img
    """
    # Perform inference for the given image.
    threshold = threshold / 100
    anomaly_map, anomaly_score = inferencer.predict(image=image, superimpose=False)
    heat_map = superimpose_anomaly_map(anomaly_map, image)
    pred_mask = compute_mask(anomaly_map, threshold)
    vis_img = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")
    return anomaly_map, anomaly_score, heat_map, pred_mask, vis_img


def get_args() -> Namespace:
    r"""Get command line arguments.

    Example:

        Example for Torch Inference.
        >>> python tools/inference/gradio_inference.py  \                                                                                     ─╯
        ...     --config ./anomalib/models/padim/config.yaml    \
        ...     --weights ./results/padim/mvtec/bottle/weights/model.ckpt   # noqa: E501    #pylint: disable=line-too-long

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a config file")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--meta_data", type=Path, required=False, help="Path to a JSON file containing the metadata.")
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=75.0,
        help="Value to threshold anomaly scores into 0-100 range",
    )
    parser.add_argument("--share", type=bool, required=False, default=False, help="Share Gradio `share_url`")

    return parser.parse_args()


def get_inferencer(config_path: Path, weight_path: Path, meta_data_path: Optional[Path] = None) -> Inferencer:
    """Parse args and open inferencer.

    Args:
        config_path (Path): Path to model configuration file or the name of the model.
        weight_path (Path): Path to model weights.
        meta_data_path (Optional[Path], optional): Metadata is required for OpenVINO models. Defaults to None.

    Raises:
        ValueError: If unsupported model weight is passed.

    Returns:
        Inferencer: Torch or OpenVINO inferencer.
    """
    config = get_configurable_parameters(config_path=config_path)

    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
    # for the openvino models.
    extension = weight_path.suffix
    inferencer: Inferencer
    module = import_module("anomalib.deploy")
    if extension in (".ckpt"):
        torch_inferencer = getattr(module, "TorchInferencer")
        inferencer = torch_inferencer(config=config, model_source=weight_path, meta_data_path=meta_data_path)

    elif extension in (".onnx", ".bin", ".xml"):
        openvino_inferencer = getattr(module, "OpenVINOInferencer")
        inferencer = openvino_inferencer(config=config, path=weight_path, meta_data_path=meta_data_path)

    else:
        raise ValueError(
            f"Model extension is not supported. Torch Inferencer exptects a .ckpt file,"
            f"OpenVINO Inferencer expects either .onnx, .bin or .xml file. Got {extension}"
        )

    return inferencer


if __name__ == "__main__":
    args = get_args()

    gradio_inferencer = get_inferencer(args.config, args.weights, args.meta_data)

    interface = gr.Interface(
        fn=lambda image, threshold: infer(image, gradio_inferencer, threshold),
        inputs=[
            gradio.inputs.Image(
                shape=None, image_mode="RGB", source="upload", tool="editor", type="numpy", label="Image"
            ),
            gradio.inputs.Slider(default=args.threshold, label="threshold", optional=False),
        ],
        outputs=[
            gradio.outputs.Image(type="numpy", label="Anomaly Map"),
            gradio.outputs.Textbox(type="number", label="Anomaly Score"),
            gradio.outputs.Image(type="numpy", label="Predicted Heat Map"),
            gradio.outputs.Image(type="numpy", label="Predicted Mask"),
            gradio.outputs.Image(type="numpy", label="Segmentation Result"),
        ],
        title="Anomalib",
        description="Anomalib Gradio",
    )

    interface.launch(share=args.share)
