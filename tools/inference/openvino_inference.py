"""Anomalib Inferencer Script.

This script performs inference by reading a model config file from
command line, and show the visualization results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings
from argparse import ArgumentParser
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from anomalib.deploy import OpenVINOInferencer

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
LABEL_MAPPING = {0: "Normal", 1: "Anomaly"}


class Visualizer:
    """Visualizer for the inference results.

    Args:
        visualization_mode (str, optional): Visualization mode. One of `simple` and `full`. Defaults to "simple".
    """

    def __init__(self, visualization_mode: str = "simple") -> None:
        self.visualization_mode = visualization_mode

    @staticmethod
    def superimpose_anomaly_map(anomaly_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Superimpose anomaly map on the input image.

        Args:
            anomaly_map (np.ndarray): Anomaly map.
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Superimposed image.
        """
        # convert to color map
        anomaly_map *= 255
        anomaly_map = anomaly_map.astype(np.uint8)
        anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_BGR2RGB)

        superimposed_map = cv2.addWeighted(image, 0.5, anomaly_map, 0.5, 0)
        return superimposed_map

    def visualize(
        self,
        image: np.ndarray,
        predictions: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Visualize the inference results.

        Args:
            image (np.ndarray): Input image. Currently assumes single image in batch.
            predictions (dict[str, np.ndarray]): Inference results

        Returns:
            np.ndarray: Visualized image.
        """
        image = image.squeeze(0)
        image = image.transpose(1, 2, 0).astype(np.uint8)

        result: np.ndarray
        if self.visualization_mode == "simple":
            result = self.visualization_simple(image, predictions)
        elif self.visualization_mode == "full":
            result = self.visualization_full(image, predictions)
        else:
            raise ValueError(f"Unsupported visualization mode: {self.visualization_mode}")

        return result

    @staticmethod
    def save(image: np.ndarray, input_path: Path, output_path: Path) -> None:
        """Save the visualized image.

        Args:
            image (np.ndarray): Visualized image.
            input_path (Path): Input image path.
            output_path (Path): Output image path.
        """
        if output_path.is_dir():
            file_path = output_path / input_path.parent.name / input_path.name
        else:
            file_path = output_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.is_file():
            warnings.warn(f"File {file_path} already exists. Overwriting file.")

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(file_path), image)

    def visualization_simple(self, image: np.ndarray, outputs: dict[str, np.ndarray]) -> np.ndarray:
        """Simple visualization of the inference results.

        Results are superimposed over each other.


        Args:
            image (np.ndarray): Input image.
            outputs (dict[str, np.ndarray]): Inference results.

        Returns:
            np.ndarray: Visualized image.
        """
        if "anomaly_map" in outputs:
            image = self.superimpose_anomaly_map(outputs["anomaly_map"], image)
        if "pred_mask" in outputs:
            self.superimpose_pred_mask(image, outputs["pred_mask"])
        if "pred_label" in outputs:
            pred_label = LABEL_MAPPING[outputs["pred_label"].item()]
            pred_label += f" ({outputs.get('pred_score',None):.2f})"
            image = cv2.putText(
                image,
                pred_label,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if "pred_boxes" in outputs:
            self.superimpose_pred_boxes(image, outputs["pred_boxes"])
        return image

    def visualization_full(self, image: np.ndarray, outputs: dict[str, np.ndarray]) -> np.ndarray:
        """Full visualization of the inference results.

        Results are shown in separate images.

        Args:
            image (np.ndarray): Input image.
            outputs (dict[str, np.ndarray]): Inference results.

        Returns:
            np.ndarray: Visualized image.
        """
        output_keys = [
            key for key in outputs if key not in ("pred_label", "pred_score", "box_labels", "pred_boxes_per_image")
        ]
        num_images = len(output_keys) + 1
        plt.axis("off")
        fig, ax = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

        for axis in ax:
            axis.axes.xaxis.set_visible(False)
            axis.axes.yaxis.set_visible(False)

        ax[0].imshow(image)
        ax[0].set_title("Input Image")
        for i, key in enumerate(output_keys):
            value = outputs[key]
            if key == "anomaly_map":
                value = self.superimpose_anomaly_map(anomaly_map=value, image=image.copy())
            elif key == "pred_mask":
                value = self.superimpose_pred_mask(image=image.copy(), mask=value)
            elif key == "pred_boxes":
                value = self.superimpose_pred_boxes(image.copy(), value)
            ax[i + 1].imshow(value)
            ax[i + 1].set_title(" ".join(key.split("_")).title())
        label = ""
        if "pred_label" in outputs:
            label = LABEL_MAPPING[outputs["pred_label"].item()]
            label += f" (Pred Score: {outputs.get('pred_score',None):.2f})"
        fig.suptitle(label)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img

    @staticmethod
    def superimpose_pred_boxes(image: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
        """Superimpose predicted boxes on image.

        Args:
            image (np.ndarray): Input image.
            pred_boxes (np.ndarray): Predicted boxes.

        Returns:
            np.ndarray: Image with superimposed boxes.
        """
        for box in pred_boxes:
            if len(box) == 4:
                image = cv2.rectangle(
                    image,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (255, 0, 0),
                    2,
                )
        return image

    @staticmethod
    def superimpose_pred_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Superimpose predicted mask on image.

        Args:
            image (np.ndarray): Input image.
            mask (np.ndarray): Predicted mask.

        Returns:
            np.ndarray: Image with superimposed mask.
        """
        if mask.max() <= 1.0:
            mask *= 255
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, contours, -1, (255, 0, 0), 5)
        return image

    @staticmethod
    def show(image: np.ndarray):
        """Show image."""
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Predictions", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class Runner:
    """Runs inference.

    Args:
        weights (Path | str): Path to model weights.
        device (str): Device to run inference on.
        visualization_mode (str): Visualization mode. One of "full", "simple".
        inputs (Path | str): Path to input image or folder.
        outputs (Path | str): Path to output folder.
        show (bool, optional): Show results in a window. Defaults to False.
    """

    def __init__(
        self,
        weights: Path | str,
        device: str,
        visualization_mode: str,
        inputs: Path | str,
        outputs: Path | str,
        show: bool = False,
    ):
        self.weights = Path(weights)
        self.device = device
        self.visualization_mode = visualization_mode
        self.inputs = inputs
        self.outputs = Path(outputs)
        self.show = show

    @staticmethod
    def get_image_filenames(path: str | Path) -> list[Path]:
        """Get image filenames.

        Args:
            path (str | Path): Path to image or image-folder.

        Returns:
            list[Path]: List of image filenames
        """
        image_filenames: list[Path] = []
        if isinstance(path, str):
            path = Path(path)
        if path.is_file() and path.suffix in IMG_EXTENSIONS:
            image_filenames = [path]
        if path.is_dir():
            image_filenames = [p for p in path.glob("**/*") if p.suffix in IMG_EXTENSIONS]
        if len(image_filenames) == 0:
            raise ValueError(f"Found 0 images in {path}")
        return image_filenames

    @staticmethod
    def read_image(filename: Path, image_size: tuple[int, int]) -> np.ndarray:
        """Read image."""
        image = cv2.imread(str(filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        image = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)  # 1 X C X H X W
        return image.astype(np.float32)

    def run(self):
        """Run the inference."""
        inferencer = OpenVINOInferencer(
            weights=self.weights,
            device=self.device,
        )
        visualizer = Visualizer(visualization_mode=self.visualization_mode)

        filenames = self.get_image_filenames(self.inputs)
        for filename in filenames:
            image = self.read_image(filename, inferencer.image_size)
            predictions = inferencer.predict(image=image)
            output = visualizer.visualize(image=image, predictions=predictions)
            if self.outputs is not None:
                visualizer.save(output, filename, self.outputs)

            if self.show:
                visualizer.show(output)


def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True, help="Path to an image to infer.")
    parser.add_argument("--output", type=Path, required=False, help="Path to save the output image.")
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Hardware device on which the model will be deployed",
        default="CPU",
        choices=["CPU", "GPU", "VPU"],
    )
    parser.add_argument(
        "--visualization_mode",
        type=str,
        required=False,
        default="simple",
        help="Visualization mode.",
        choices=["full", "simple"],
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    runner = Runner(args.weights, args.device, args.visualization_mode, args.input, args.output, args.show)
    runner.run()
