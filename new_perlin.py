import logging
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # Use F for torch.nn.functional
import torchvision.transforms.v2.functional as F_v2  # Use F_v2 for torchvision functional
from PIL import Image
from torchvision import io
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF
from tqdm import tqdm

from anomalib.data.transforms import MultiRandomChoice
from anomalib.data.utils.augmenter import Augmenter
from anomalib.data.utils.generators.perlin import random_2d_perlin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def nextpow2(value: int) -> int:
    """Return the smallest power of 2 greater than or equal to the input value."""
    return 2 ** (math.ceil(math.log(value, 2)))


class DRAEMAugmenter(v2.Transform):
    """Torchvision v2 implementation of DRAEM augmentations with corrected transform parameters."""

    def __init__(
        self,
        anomaly_source_path: str | None = None,
        p_anomalous: float = 0.5,
        beta: float | tuple[float, float] = (0.2, 1.0),
    ) -> None:
        super().__init__()
        self.p_anomalous = p_anomalous
        self.beta = beta

        # Load anomaly source paths
        self.anomaly_source_paths: list[Path] = []
        if anomaly_source_path is not None:
            for img_ext in IMG_EXTENSIONS:
                self.anomaly_source_paths.extend(Path(anomaly_source_path).rglob("*" + img_ext))

        # Corrected augmentations to match imgaug behavior
        self.augmenters = MultiRandomChoice(
            transforms=[
                # Using ColorJitter's contrast with adjusted range to match gamma contrast
                v2.ColorJitter(contrast=(0.5, 2.0)),  # Adjusted range
                # MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30))
                v2.Compose([
                    # Split into two operations to match imgaug behavior
                    v2.Lambda(lambda x: x * torch.empty(1).uniform_(0.8, 1.2).item()),
                    v2.Lambda(lambda x: torch.clamp(x + torch.empty(1).uniform_(-30 / 255, 30 / 255).item(), 0, 1)),
                ]),
                # EnhanceSharpness()
                v2.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
                # AddToHueAndSaturation((-50, 50), per_channel=True)
                v2.Compose([
                    # Adjust hue and saturation separately to match per-channel behavior
                    v2.ColorJitter(
                        hue=[-50 / 360, 50 / 360],  # Convert degree range to [0,1]
                        saturation=[0.5, 1.5],  # Adjusted range for saturation
                    ),
                ]),
                # Solarize(0.5, threshold=(32, 128))
                v2.RandomSolarize(threshold=torch.empty(1).uniform_(32 / 255, 128 / 255).item(), p=1.0),
                # Posterize()
                v2.RandomPosterize(bits=4, p=1.0),
                # Invert()
                v2.RandomInvert(p=1.0),
                # Autocontrast() - Custom implementation to match imgaug
                v2.Lambda(lambda x: TF.autocontrast(x)),
                # Equalize()
                v2.RandomEqualize(p=1.0),
                # Affine(rotate=(-45, 45))
                v2.RandomAffine(degrees=(-45, 45), interpolation=v2.InterpolationMode.BILINEAR, fill=0),
            ],
            probabilities=None,  # Equal probabilities
            num_transforms=3,  # Always pick 3 transforms
            fixed_num_transforms=True,
        )

        # Rotation transform for perlin noise
        self.rot = v2.RandomAffine(degrees=(-90, 90), interpolation=v2.InterpolationMode.BILINEAR, fill=0)

    def generate_perturbation(
        self,
        height: int,
        width: int,
        device: torch.device,
        anomaly_source_path: Path | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate perturbed image and mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - perturbation: The augmented image tensor [H, W, C]
                - mask: Binary mask tensor [H, W, 1]
        """
        # 1. Generate perlin noise base
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** np.random.default_rng().integers(min_perlin_scale, perlin_scale)
        perlin_scaley = 2 ** np.random.default_rng().integers(min_perlin_scale, perlin_scale)

        # Get perlin noise array
        perlin_noise = random_2d_perlin((nextpow2(height), nextpow2(width)), (perlin_scalex, perlin_scaley))[
            :height,
            :width,
        ]

        # 2. Create rotated noise pattern
        perlin_noise = torch.from_numpy(perlin_noise).to(device).unsqueeze(0)  # [1, H, W]
        perlin_noise = self.rot(perlin_noise).squeeze(0)  # [H, W]

        # 3. Generate binary mask from perlin noise
        mask = torch.where(
            perlin_noise > 0.5,
            torch.ones_like(perlin_noise, device=device),
            torch.zeros_like(perlin_noise, device=device),
        ).unsqueeze(-1)  # [H, W, 1]

        # 4. Generate anomaly source image
        if anomaly_source_path:
            # Use provided image as source
            anomaly_source_img = (
                io.read_image(path=str(anomaly_source_path), mode=io.ImageReadMode.RGB).float().to(device) / 255.0
            )

            if anomaly_source_img.shape[-2:] != (height, width):
                anomaly_source_img = F_v2.resize(anomaly_source_img, [height, width], antialias=True)

            anomaly_source_img = anomaly_source_img.permute(1, 2, 0)  # [H, W, C]
        else:
            # Use perlin noise as source
            anomaly_source_img = perlin_noise.unsqueeze(-1).repeat(1, 1, 3)  # [H, W, C]
            anomaly_source_img = (anomaly_source_img * 0.5) + 0.25  # Adjust intensity range

        # 5. Apply augmentations to source image
        anomaly_augmented = self.augmenters(anomaly_source_img.permute(2, 0, 1))  # [C, H, W]
        anomaly_augmented = anomaly_augmented.permute(1, 2, 0)  # [H, W, C]

        # 6. Create final perturbation by applying mask
        perturbation = anomaly_augmented * mask

        return perturbation, mask

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation using the mask."""
        device = img.device
        c, h, w = img.shape

        if torch.rand(1, device=device) > self.p_anomalous:
            return img, torch.zeros((1, h, w), device=device)

        anomaly_source_path = (
            random.sample(self.anomaly_source_paths, 1)[0] if len(self.anomaly_source_paths) > 0 else None
        )

        # Generate perturbation and mask
        perturbation, mask = self.generate_perturbation(h, w, device, anomaly_source_path)

        # Adjust dimensions
        perturbation = perturbation.permute(2, 0, 1)  # [C, H, W]
        mask = mask.permute(2, 0, 1)  # [1, H, W]

        # Calculate beta
        if isinstance(self.beta, float):
            beta = self.beta
        elif isinstance(self.beta, tuple):
            beta = torch.rand(1, device=device) * (self.beta[1] - self.beta[0]) + self.beta[0]
            beta = beta.view(-1, 1, 1).expand_as(img)
        else:
            raise TypeError("Beta must be either float or tuple of floats")

        # Apply perturbation with beta blending
        augmented_img = img * (1 - mask) + beta * perturbation + (1 - beta) * img * mask

        return augmented_img, mask


class DRAEMComparer:
    """Compare imgaug and torchvision implementations of DRAEM augmentations."""

    def __init__(
        self,
        test_images_dir: str,
        anomaly_source_dir: str | None = None,
        num_samples: int = 100,
        seed: int = 42,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.test_images_dir = Path(test_images_dir)
        if not self.test_images_dir.exists():
            raise ValueError(f"Test images directory does not exist: {test_images_dir}")

        self.anomaly_source_dir = Path(anomaly_source_dir) if anomaly_source_dir else None
        if self.anomaly_source_dir and not self.anomaly_source_dir.exists():
            raise ValueError(f"Anomaly source directory does not exist: {anomaly_source_dir}")

        self.num_samples = num_samples
        self.device = device

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize both augmenters
        self.imgaug_augmenter = Augmenter(anomaly_source_path=anomaly_source_dir, p_anomalous=1.0, beta=0.5)

        self.torchvision_augmenter = DRAEMAugmenter(anomaly_source_path=anomaly_source_dir, p_anomalous=1.0, beta=0.5)

    def load_test_images(self) -> list[torch.Tensor]:
        """Load test images from directory."""
        images = []
        image_paths = []

        # Collect all image paths
        for ext in IMG_EXTENSIONS:
            image_paths.extend(self.test_images_dir.glob(f"*{ext}"))
            image_paths.extend(self.test_images_dir.glob(f"*{ext.upper()}"))

        if not image_paths:
            raise ValueError(f"No valid images found in {self.test_images_dir}")

        # Limit to num_samples
        image_paths = image_paths[: self.num_samples]
        logger.info(f"Found {len(image_paths)} images to process")

        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        for img_path in tqdm(image_paths, desc="Loading images"):
            img = transform(Image.open(img_path).convert("RGB"))
            images.append(img)

        if not images:
            raise ValueError("No images were successfully loaded")

        return images

    def compare_augmentations(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run both augmentations and return results with proper augmentation application."""
        device = image.device
        dtype = torch.float32
        image = image.to(dtype=dtype)

        _, h, w = image.shape

        # imgaug version
        img_np = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        perturbation, mask_imgaug = self.imgaug_augmenter.generate_perturbation(img_np.shape[0], img_np.shape[1])

        # Convert perturbation and mask to torch tensors
        perturbation = torch.from_numpy(perturbation).to(device=device, dtype=dtype).permute(2, 0, 1) / 255.0
        mask_imgaug = torch.from_numpy(mask_imgaug).to(device=device, dtype=dtype).permute(2, 0, 1)

        # Apply the perturbation to the original image using the same blending formula
        if isinstance(self.imgaug_augmenter.beta, float):
            beta = self.imgaug_augmenter.beta
        else:
            beta = random.uniform(self.imgaug_augmenter.beta[0], self.imgaug_augmenter.beta[1])

        # Apply perturbation to original image
        aug_imgaug = image * (1 - mask_imgaug) + beta * perturbation + (1 - beta) * image * mask_imgaug

        if aug_imgaug.shape[-2:] != (h, w):
            aug_imgaug = F_v2.resize(aug_imgaug, [h, w], antialias=True)
            mask_imgaug = F_v2.resize(mask_imgaug, [h, w], antialias=True)

        # Debug imgaug outputs
        logger.debug(
            f"Augmented imgaug - shape: {aug_imgaug.shape}, dtype: {aug_imgaug.dtype}, "
            f"min: {aug_imgaug.min():.3f}, max: {aug_imgaug.max():.3f}",
        )
        logger.debug(
            f"Mask imgaug - shape: {mask_imgaug.shape}, dtype: {mask_imgaug.dtype}, "
            f"min: {mask_imgaug.min():.3f}, max: {mask_imgaug.max():.3f}",
        )

        # torchvision version
        aug_torch, mask_torch = self.torchvision_augmenter(image)
        aug_torch = aug_torch.to(dtype=dtype)
        mask_torch = mask_torch.to(dtype=dtype)

        # Debug torchvision outputs
        logger.debug(
            f"Augmented torch - shape: {aug_torch.shape}, dtype: {aug_torch.dtype}, "
            f"min: {aug_torch.min():.3f}, max: {aug_torch.max():.3f}",
        )
        logger.debug(
            f"Mask torch - shape: {mask_torch.shape}, dtype: {mask_torch.dtype}, "
            f"min: {mask_torch.min():.3f}, max: {mask_torch.max():.3f}",
        )

        return aug_imgaug, mask_imgaug, aug_torch, mask_torch

    def compute_metrics(
        self,
        aug_imgaug: torch.Tensor,
        aug_torch: torch.Tensor,
        mask_imgaug: torch.Tensor,
        mask_torch: torch.Tensor,
    ) -> dict:
        """Compute comparison metrics between implementations."""
        # Ensure all inputs are on the same device and dtype
        device = aug_imgaug.device
        dtype = torch.float32  # Force float32 for all operations

        aug_imgaug = aug_imgaug.to(device=device, dtype=dtype)
        aug_torch = aug_torch.to(device=device, dtype=dtype)
        mask_imgaug = mask_imgaug.to(device=device, dtype=dtype)
        mask_torch = mask_torch.to(device=device, dtype=dtype)

        if aug_imgaug.dim() == 3:
            aug_imgaug = aug_imgaug.unsqueeze(0)
        if aug_torch.dim() == 3:
            aug_torch = aug_torch.unsqueeze(0)

        mse_img = F.mse_loss(aug_imgaug, aug_torch)
        psnr_img = 10 * torch.log10(1 / (mse_img + 1e-8))
        ssim_img = self.compute_ssim(aug_imgaug.squeeze(0), aug_torch.squeeze(0))

        mask_imgaug = mask_imgaug.float()
        mask_torch = mask_torch.float()

        if mask_imgaug.dim() == 3:
            mask_imgaug = mask_imgaug.unsqueeze(0)
        if mask_torch.dim() == 3:
            mask_torch = mask_torch.unsqueeze(0)

        mse_mask = F.mse_loss(mask_imgaug, mask_torch)
        iou = self.compute_iou(mask_imgaug > 0.5, mask_torch > 0.5)

        return {
            "Image_MSE": mse_img.item(),
            "Image_PSNR": psnr_img.item(),
            "Image_SSIM": ssim_img.item(),
            "Mask_MSE": mse_mask.item(),
            "Mask_IoU": iou.item(),
        }

    @staticmethod
    def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Compute SSIM between two images.

        Args:
            img1: Input image tensor of shape [C, H, W]
            img2: Input image tensor of shape [C, H, W]
            window_size: Size of the Gaussian window

        Returns:
            Average SSIM value across channels
        """
        # Ensure same dtype and device
        dtype = img1.dtype
        device = img1.device

        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
        if len(img2.shape) == 3:
            img2 = img2.unsqueeze(0)

        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2

        # Generate Gaussian kernel with matching dtype
        sigma = 1.5
        gauss = torch.tensor(
            [np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)],
            dtype=dtype,
        )
        gauss = gauss / gauss.sum()

        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)

        ssim_vals = []

        for c in range(img1.shape[1]):
            ch1 = img1[:, c : c + 1]
            ch2 = img2[:, c : c + 1]

            # Ensure same dtype for convolution operation
            ch1 = ch1.to(dtype=dtype)
            ch2 = ch2.to(dtype=dtype)

            mu1 = F.conv2d(ch1, kernel, padding=window_size // 2)
            mu2 = F.conv2d(ch2, kernel, padding=window_size // 2)

            mu1_sq = mu1**2
            mu2_sq = mu2**2
            mu12 = mu1 * mu2

            sigma1_sq = F.conv2d(ch1**2, kernel, padding=window_size // 2) - mu1_sq
            sigma2_sq = F.conv2d(ch2**2, kernel, padding=window_size // 2) - mu2_sq
            sigma12 = F.conv2d(ch1 * ch2, kernel, padding=window_size // 2) - mu12

            ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            ssim_vals.append(ssim_map.mean())

        return torch.stack(ssim_vals).mean()

    @staticmethod
    def compute_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two masks."""
        intersection = (mask1 & mask2).float().sum()
        union = (mask1 | mask2).float().sum()
        return intersection / (union + 1e-6)

    def visualize_results(
        self,
        original: torch.Tensor,
        aug_imgaug: torch.Tensor,
        mask_imgaug: torch.Tensor,
        aug_torch: torch.Tensor,
        mask_torch: torch.Tensor,
        metrics: dict,
    ) -> plt.Figure:
        """Create visualization of results with proper tensor normalization."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        def tensor_to_numpy(t):
            """Convert tensor to numpy array with proper normalization."""
            if t.dim() == 4:
                t = t.squeeze(0)
            if t.dim() == 2:
                t = t.unsqueeze(0)

            # Debug print for tensor stats
            logger.debug(
                f"Tensor stats before conversion - min: {t.min():.3f}, max: {t.max():.3f}, "
                f"mean: {t.mean():.3f}, shape: {t.shape}",
            )

            # Ensure values are in [0, 1] range
            if t.min() < 0 or t.max() > 1:
                logger.warning(f"Tensor values outside [0,1] range. Min: {t.min():.3f}, Max: {t.max():.3f}")
                t = t.clamp(0, 1)

            # Convert to numpy and ensure proper range
            img_np = t.detach().cpu().permute(1, 2, 0).numpy()
            logger.debug(
                f"NumPy array stats - min: {img_np.min():.3f}, max: {img_np.max():.3f}, "
                f"mean: {img_np.mean():.3f}, shape: {img_np.shape}",
            )

            return img_np

        def visualize_tensor(ax, tensor, title, is_mask=False):
            """Helper function to visualize a tensor with proper error handling."""
            try:
                if is_mask:
                    # For masks, ensure single channel and proper normalization
                    mask = tensor.squeeze()  # Remove extra dimensions
                    if mask.dim() == 3 and mask.shape[0] == 1:
                        mask = mask.squeeze(0)
                    mask_np = mask.detach().cpu().numpy()
                    im = ax.imshow(mask_np, cmap="gray", vmin=0, vmax=1)
                else:
                    # For RGB images
                    img_np = tensor_to_numpy(tensor)
                    im = ax.imshow(img_np)

                ax.set_title(title)
                return im
            except Exception as e:
                logger.error(f"Error visualizing {title}: {e!s}")
                logger.error(
                    f"Tensor stats - shape: {tensor.shape}, dtype: {tensor.dtype}, "
                    f"min: {tensor.min():.3f}, max: {tensor.max():.3f}",
                )
                ax.text(0.5, 0.5, f"Error: {e!s}", ha="center", va="center")

        # Print debug information for input tensors
        for name, tensor in [
            ("Original", original),
            ("Aug ImgAug", aug_imgaug),
            ("Mask ImgAug", mask_imgaug),
            ("Aug Torch", aug_torch),
            ("Mask Torch", mask_torch),
        ]:
            logger.debug(
                f"{name} tensor - shape: {tensor.shape}, dtype: {tensor.dtype}, "
                f"min: {tensor.min():.3f}, max: {tensor.max():.3f}, "
                f"mean: {tensor.mean():.3f}",
            )

        # Visualize images with error handling
        visualize_tensor(axes[0, 0], original, "Original")
        visualize_tensor(axes[0, 1], aug_imgaug, "imgaug Augmented")
        visualize_tensor(axes[0, 2], mask_imgaug, "imgaug Mask", is_mask=True)
        visualize_tensor(axes[1, 1], aug_torch, "torchvision Augmented")
        visualize_tensor(axes[1, 2], mask_torch, "torchvision Mask", is_mask=True)

        # Difference visualization
        diff = torch.abs(aug_imgaug - aug_torch)
        visualize_tensor(axes[1, 0], diff, "Difference")

        # Add metrics text with background for better visibility
        metrics_text = "\n".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        fig.text(
            0.02,
            0.02,
            metrics_text,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=5),
        )

        for ax in axes.flat:
            ax.axis("off")

        plt.tight_layout()
        return fig

    def run_comparison(self, save_dir: str | None = None) -> dict:
        """Run comparison with consistent data types."""
        save_dir = Path(save_dir) if save_dir else None
        if save_dir:
            save_dir.mkdir(exist_ok=True, parents=True)

        # Set default dtype for torch operations
        torch.set_default_dtype(torch.float32)

        test_images = self.load_test_images()
        all_metrics = []

        for idx, image in enumerate(tqdm(test_images, desc="Comparing implementations")):
            # Ensure consistent dtype
            image = image.to(dtype=torch.float32, device=self.device)

            aug_imgaug, mask_imgaug, aug_torch, mask_torch = self.compare_augmentations(image)
            metrics = self.compute_metrics(aug_imgaug, aug_torch, mask_imgaug, mask_torch)
            all_metrics.append(metrics)

            if save_dir:
                fig = self.visualize_results(image, aug_imgaug, mask_imgaug, aug_torch, mask_torch, metrics)
                fig.savefig(save_dir / f"comparison_{idx:04d}.png")
                plt.close(fig)

        if not all_metrics:
            raise ValueError("No comparisons were successfully completed")

        # Compute aggregate statistics
        agg_metrics = {
            metric: {
                "mean": np.mean([m[metric] for m in all_metrics]),
                "std": np.std([m[metric] for m in all_metrics]),
                "min": np.min([m[metric] for m in all_metrics]),
                "max": np.max([m[metric] for m in all_metrics]),
            }
            for metric in all_metrics[0].keys()
        }

        # Save aggregate metrics if save_dir is provided
        if save_dir:
            metrics_path = save_dir / "aggregate_metrics.txt"
            with open(metrics_path, "w") as f:
                for metric, stats in agg_metrics.items():
                    f.write(f"\n{metric}:\n")
                    for stat_name, value in stats.items():
                        f.write(f"  {stat_name}: {value:.6f}\n")
            logger.info(f"Saved aggregate metrics to {metrics_path}")

        return agg_metrics


if __name__ == "__main__":
    comparer = DRAEMComparer(
        test_images_dir="./datasets/MVTec/bottle/test/good",
        anomaly_source_dir="./datasets/MVTec/bottle/test/good",
        num_samples=50,
        seed=42,
    )

    metrics = comparer.run_comparison(save_dir="comparison_results")
    print(metrics)
