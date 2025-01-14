"""Explanation visualization generator for model interpretability.

This module provides utilities for visualizing model explanations and
interpretability results. The key components include:

    - Text-based explanations rendered on images
    - Label visualization for model decisions
    - Combined visualization with original image and explanation

Example:
    >>> from anomalib.utils.visualization import ExplanationVisualizer
    >>> # Create visualizer
    >>> visualizer = ExplanationVisualizer()
    >>> # Generate visualization
    >>> results = visualizer.generate(
    ...     outputs={
    ...         "image": images,
    ...         "explanation": explanations,
    ...         "image_path": paths
    ...     }
    ... )

Note:
    This is a temporary visualizer that will be replaced with an enhanced
    version in a future release.

The module ensures consistent visualization of model explanations across
different interpretability approaches.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import BaseVisualizer, GeneratorResult, VisualizationStep


class ExplanationVisualizer(BaseVisualizer):
    """Explanation visualization generator."""

    def __init__(self) -> None:
        super().__init__(visualize_on=VisualizationStep.BATCH)
        self.padding = 3
        self.font = ImageFont.load_default(size=16)

    def generate(self, **kwargs) -> Iterator[GeneratorResult]:
        """Generate images and return them as an iterator."""
        outputs = kwargs.get("outputs", None)
        if outputs is None:
            msg = "Outputs must be provided to generate images."
            raise ValueError(msg)
        return self._visualize_batch(outputs)

    def _visualize_batch(self, batch: dict) -> Iterator[GeneratorResult]:
        """Visualize batch of images."""
        batch_size = batch["image"].shape[0]
        height, width = batch["image"].shape[-2:]
        for i in range(batch_size):
            image = batch["image"][i]
            explanation = batch["explanation"][i]
            file_name = Path(batch["image_path"][i])
            image = Image.open(file_name)
            image = image.resize((width, height))
            image = self._draw_image(width, height, image=image, explanation=explanation)
            yield GeneratorResult(image=image, file_name=file_name)

    def _draw_image(self, width: int, height: int, image: Image, explanation: str) -> np.ndarray:
        text_canvas: Image = self._get_explanation_image(width, height, image, explanation)
        label_canvas: Image = self._get_label_image(explanation)

        final_width = max(text_canvas.size[0], width)
        final_height = height + text_canvas.size[1]
        combined_image = Image.new("RGB", (final_width, final_height), (255, 255, 255))
        combined_image.paste(image, (self.padding, 0))
        combined_image.paste(label_canvas, (10, 10))
        combined_image.paste(text_canvas, (0, height))
        return np.array(combined_image)

    def _get_label_image(self, explanation: str) -> Image:
        # Draw label
        # Can't use  pred_labels as it is computed from the pred_scores using image_threshold. It gives incorrect value.
        # So, using explanation. This will probably change with the new design.
        label = "Anomalous" if explanation.startswith("Y") else "Normal"
        label_color = "red" if label == "Anomalous" else "green"
        label_canvas = Image.new("RGB", (100, 20), color=label_color)
        draw = ImageDraw.Draw(label_canvas)
        draw.text((0, 0), label, font=self.font, fill="white", align="center")
        return label_canvas

    def _get_explanation_image(self, width: int, height: int, image: Image, explanation: str) -> Image:
        # compute wrap width
        text_canvas = Image.new("RGB", (width, height), color="white")
        dummy_image = ImageDraw.Draw(image)
        text_bbox = dummy_image.textbbox((0, 0), explanation, font=self.font, align="center")
        text_canvas_width = text_bbox[2] - text_bbox[0] + self.padding

        # split lines based on the width
        lines = list(explanation.split("\n"))
        line_with_max_len = max(lines, key=len)
        new_width = int(width * len(line_with_max_len) // text_canvas_width)

        # wrap text based on the new width
        lines = []
        current_line: list[str] = []
        for word in explanation.split(" "):
            test_line = " ".join([*current_line, word])
            if len(test_line) <= new_width:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        lines.append(" ".join(current_line))
        wrapped_lines = "\n".join(lines)

        # recompute height
        dummy_image = Image.new("RGB", (new_width, height), color="white")
        draw = ImageDraw.Draw(dummy_image)
        text_bbox = draw.textbbox((0, 0), wrapped_lines, font=self.font, align="center")
        new_width = int(text_bbox[2] - text_bbox[0] + self.padding)
        new_height = int(text_bbox[3] - text_bbox[1] + self.padding)

        # Final text image
        text_canvas = Image.new("RGB", (new_width, new_height), color="white")
        draw = ImageDraw.Draw(text_canvas)
        draw.text((self.padding // 2, 0), wrapped_lines, font=self.font, fill="black", align="center")
        return text_canvas
