```{eval-rst}
:orphan:
```

# Visualization in Anomalib

```{warning}
The visualization module is currently experimental. The API and functionality may change in future releases without following semantic versioning. Use with caution in production environments.

Key points:
- API may change without notice
- Some features might be unstable
- Default configurations might be adjusted
- New visualization methods may be added or removed
```

This guide explains how visualization works in Anomalib, its components, and how to use them effectively.

## Overview

Anomalib provides a powerful visualization system that:

- Visualizes anomaly detection results (images, masks, anomaly maps)
- Supports both classification and segmentation results
- Offers customizable visualization options
- Maintains consistent output formats

The visualization system consists of:

1. `ImageVisualizer` - A Lightning Callback for automatic visualization during training/testing
2. `visualize_image_item` - Core function for visualizing `ImageItem` objects
3. Utility functions for specific visualization tasks (masks, anomaly maps, etc.)

## Basic Usage

### Using the Visualizer Callback

The `ImageVisualizer` is a callback that automatically visualizes results during test time:

```python
from anomalib.visualization import ImageVisualizer
from anomalib.engine import Engine
from anomalib.models import Patchcore

# Create visualizer with default settings
visualizer = ImageVisualizer()

# Create model with visualizer
model = Patchcore(
    visualizer=visualizer  # Pass visualizer to the model
)

# Create engine
engine = Engine()

# The visualizer will automatically create visualizations
# during test_step and predict_step
engine.test(model, datamodule)
```

### Direct Visualization

For direct visualization of `ImageItem` objects, use `visualize_image_item`:

```python
from anomalib.visualization.image.item_visualizer import visualize_image_item
from anomalib.data import ImageItem
from PIL import Image
import torch
from torchvision.io import read_image

# Create sample data
image_path = "./datasets/MVTecAD/bottle/test/broken_large/000.png"
mask_path = "./datasets/MVTecAD/bottle/ground_truth/broken_large/000_mask.png"
image = read_image(image_path)
mask = read_image(mask_path)

# Create an ImageItem
item = ImageItem(
    image_path=image_path,
    mask_path=mask_path,
    image=image,
    gt_mask=mask,
)

# Generate visualization
vis_result = visualize_image_item(item, fields=["image", "gt_mask"])
```

## Visualization Components

### 1. Anomaly Maps

Visualize anomaly heatmaps:

```python
from anomalib.visualization import visualize_anomaly_map
import torch

# Create sample anomaly map
anomaly_map = torch.rand(256, 256)

# Visualize with default settings
vis = visualize_anomaly_map(anomaly_map)

# Customize visualization
vis = visualize_anomaly_map(
    anomaly_map,
    colormap=True,      # Apply colormap
    normalize=True      # Normalize values to [0, 255]
)
```

### 2. Segmentation Masks

Visualize ground truth and predicted masks:

```python
import torch

from anomalib.visualization.image.functional import visualize_gt_mask, visualize_pred_mask

# Create sample mask
mask = torch.zeros((256, 256))
mask[100:150, 100:150] = 1

# Visualize ground truth mask
gt_vis = visualize_gt_mask(
    mask,
    mode="contour",            # Draw mask boundaries
    color=(0, 255, 0),        # Green color
    alpha=0.7                  # Opacity
)

# Visualize prediction mask
pred_vis = visualize_pred_mask(
    mask,
    mode="fill",              # Fill mask regions
    color=(255, 0, 0),        # Red color
    alpha=0.5,                # Opacity
)
```

## Advanced Usage

### 1. Custom Visualization Configurations

Configure visualization settings and pass to the model:

```python
from anomalib.visualization import ImageVisualizer

# Custom visualization settings
visualizer = ImageVisualizer(
    fields_config={
        "image": {},  # Default image display
        "anomaly_map": {
            "colormap": True,
            "normalize": True
        },
        "pred_mask": {
            "mode": "contour",
            "color": (255, 0, 0),
            "alpha": 0.7
        },
        "gt_mask": {
            "mode": "contour",
            "color": (0, 255, 0),
            "alpha": 0.7
        }
    }
)

# Pass visualizer to the model
model = Patchcore(visualizer=visualizer)
```

### 2. Direct Visualization with Custom Settings

For more control over visualization, use `visualize_image_item` directly:

```python
from anomalib.visualization.image.item_visualizer import visualize_image_item

# Customize which fields to visualize
result = visualize_image_item(
    item,
    fields=["image", "anomaly_map"],
    fields_config={
        "anomaly_map": {"colormap": True, "normalize": True}
    }
)

# Create overlays
result = visualize_image_item(
    item,
    overlay_fields=[("image", ["gt_mask", "pred_mask"])],
    overlay_fields_config={
        "gt_mask": {"mode": "contour", "color": (0, 255, 0), "alpha": 0.7},
        "pred_mask": {"mode": "fill", "color": (255, 0, 0), "alpha": 0.3}
    }
)
```

## Best Practices

1. **Automatic Visualization During Training/Testing**:

   ```python
   # Configure visualization as part of the model
   visualizer = ImageVisualizer(
       fields_config={"anomaly_map": {"normalize": True}}
   )
   model = Patchcore(visualizer=visualizer)
   engine = Engine()
   engine.test(model, datamodule)
   ```

2. **Custom Visualization Pipeline**:

   ```python
   # Create a custom visualization pipeline
   def create_visualization_pipeline(datamodule):
       visualizer = ImageVisualizer()
       model = Patchcore(visualizer=visualizer)
       engine = Engine()

       # Visualizations will be automatically generated
       # during test/predict steps
       engine.test(model, datamodule)
   ```

3. **Manual Batch Processing**:

   ```python
   from anomalib.visualization.image.item_visualizer import visualize_image_item

   def process_batch(batch_items):
       visualizations = []
       for item in batch_items:
           vis = visualize_image_item(
               item,
               fields=["image", "anomaly_map"],
               fields_config={"anomaly_map": {"normalize": True}}
           )
           visualizations.append(vis)
       return visualizations
   ```

## Common Pitfalls

1. **Callback Configuration**:

   ```python
   # Wrong: Trying to call visualize directly
   visualizer = ImageVisualizer()
   visualizer.visualize(item)  # This won't work!

   # Correct: Use as a callback through the model or use visualize_image_item directly
   from anomalib.visualization.image.item_visualizer import visualize_image_item
   result = visualize_image_item(item)
   ```

2. **Memory Management**:

   ```python
   # Wrong: Keeping all visualizations in memory
   visualizations = []
   for batch in test_dataloader:
       for item in batch:
           vis = visualize_image_item(item)
           visualizations.append(vis)  # Memory accumulates

   # Better: Process and save immediately
   for batch in test_dataloader:
       for item in batch:
           vis = visualize_image_item(item)
           vis.save(f"vis_{item.image_path.stem}.png")
           del vis  # Clear memory
   ```

3. **Visualization Settings**:

   ```python
   # Wrong: Inconsistent settings across visualizations
   vis1 = visualize_image_item(item1, fields_config={"anomaly_map": {"normalize": True}})
   vis2 = visualize_image_item(item2, fields_config={"anomaly_map": {"normalize": False}})

   # Better: Use consistent settings
   config = {
       "anomaly_map": {"normalize": True},
       "pred_mask": {"mode": "contour", "alpha": 0.7}
   }
   vis1 = visualize_image_item(item1, fields_config=config)
   vis2 = visualize_image_item(item2, fields_config=config)
   ```

```{seealso}
For more information:
- {doc}`AnomalibModule Documentation <../../reference/models/base>`
- {doc}`Metrics Guide <../metrics/index>`
```
