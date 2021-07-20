"""
Anomaly Map Generator
"""

import cv2
import numpy as np
from skimage import morphology
from sklearn.metrics import precision_recall_curve


class BaseAnomalyMapGenerator:
    """
    BaseAnomalyMapGenerator
    """

    def __init__(self, alpha: float = 0.4, gamma: int = 0, sigma: int = 4):
        self.sigma = sigma
        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.gamma = gamma

    @staticmethod
    def compute_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
        """Compute anomaly color heatmap

        Args:
          anomaly_map: Final anomaly map computed by the distance metric.
          anomaly_map: np.ndarray:

        Returns:
          Anomaly heatmap via Jet Color Map.

        """
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
        anomaly_map = anomaly_map * 255
        anomaly_map = anomaly_map.astype(np.uint8)

        heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        return heatmap

    def apply_heatmap_on_image(self, anomaly_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Apply anomaly heatmap on input test image.

        Args:
          anomaly_map: Anomaly color map
          image: Input test image
          anomaly_map: np.ndarray:
          image: np.ndarray:

        Returns:
          Output image, where anomaly color map is blended on top of the input image.

        """

        heatmap = self.compute_heatmap(anomaly_map)
        heatmap_on_image = cv2.addWeighted(heatmap, self.alpha, image, self.beta, self.gamma)
        heatmap_on_image = cv2.cvtColor(heatmap_on_image, cv2.COLOR_BGR2RGB)
        return heatmap_on_image

    @staticmethod
    def compute_adaptive_threshold(true_masks, anomaly_map):
        """Compute adaptive threshold, based on the f1 metric of the true and predicted anomaly masks.

        Args:
          true_masks: Ground-truth anomaly mask showing the location of anomalies.
          anomaly_map: Anomaly map that is predicted by the model.

        Returns:
          Threshold value based on the best f1 score.

        """

        precision, recall, thresholds = precision_recall_curve(true_masks.flatten(), anomaly_map.flatten())
        numerator = 2 * precision * recall
        denominator = precision + recall
        f1_score = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
        threshold = thresholds[np.argmax(f1_score)]
        max_f1 = np.max(f1_score)

        return threshold, max_f1

    @staticmethod
    def compute_mask(anomaly_map: np.ndarray, threshold: float, kernel_size: int = 4) -> np.ndarray:
        """Compute anomaly mask via thresholding the predicted anomaly map.

        Args:
          anomaly_map: Anomaly map predicted via the model
          threshold: Value to threshold anomaly scores into 0-1 range.
          kernel_size: Value to apply morphological operations to the predicted mask
          anomaly_map: np.ndarray:
          threshold: float:
          kernel_size: int:  (Default value = 4)

        Returns:
          Predicted anomaly mask

        """

        mask = np.zeros_like(anomaly_map).astype(np.uint8)
        mask[anomaly_map > threshold] = 1

        kernel = morphology.disk(kernel_size)
        mask = morphology.opening(mask, kernel)

        mask *= 255

        return mask
