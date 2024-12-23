```{eval-rst}
:orphan:
```

# Pre-processing in Anomalib

This guide explains how pre-processing works in Anomalib, its integration with models, and how to create custom pre-processors.

## Overview

Pre-processing in Anomalib is designed to encapsulate model-specific transforms (like input size and normalization) within the model graph itself. This design ensures that during deployment:

- Pre-processing is part of the exported model (ONNX, OpenVINO)
- Users don't need to manually resize or normalize inputs
- Edge deployment is simplified with automatic pre-processing

The `PreProcessor` class serves two roles:

1. A PyTorch Module for transform application that gets exported with the model
2. A Lightning Callback for managing stage-specific pre-processing steps (e.g, training, inference)

For example, when a model is exported:

```python
# During training/development
model = Patchcore(
    pre_processor=PreProcessor(
        transform=Compose([
            Resize((256, 256)),
            Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
        ])
    )
)

# Export model with pre-processing included
model.export("model.onnx")

# During deployment - no manual pre-processing needed
deployed_model = onnxruntime.InferenceSession("model.onnx")
raw_image = cv2.imread("test.jpg")  # Any size, unnormalized
prediction = deployed_model.run(None, {"input": raw_image})
```

## Basic Usage

```python
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Resize, Compose, ToTensor

# Create a simple pre-processor
pre_processor = PreProcessor(
    transform=Compose([
        Resize((256, 256)),
        ToTensor()
    ])
)

# Apply to a batch
transformed_batch = pre_processor(batch)
```

## Stage-Specific Transforms

You can specify different transforms for training, validation, and testing:

```python
from torchvision.transforms.v2 import RandomHorizontalFlip

pre_processor = PreProcessor(
    train_transform=Compose([
        Resize((256, 256)),
        RandomHorizontalFlip(),
        ToTensor()
    ]),
    val_transform=Compose([
        Resize((256, 256)),
        ToTensor()
    ]),
    test_transform=Compose([
        Resize((256, 256)),
        ToTensor()
    ])
)
```

## Integration with Models

The pre-processor is automatically integrated into Anomalib models through the `AnomalibModule` base class:

```python
from anomalib.models import Patchcore

# Model creates default pre-processor
model = Patchcore()

# Or specify custom pre-processor
model = Patchcore(
    pre_processor=PreProcessor(
        transform=Compose([
            Resize((256, 256)),
            ToTensor()
        ])
    )
)
```

### Model-Specific Pre-processing

Different models may require specific pre-processing:

1. **Patchcore**:

   ```python
   # Default pre-processing
   transform = Compose([
       Resize((256, 256)),
       ToTensor(),
       Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])  # ImageNet stats
   ])
   ```

2. **EfficientAd**:

   ```python
   # Uses specific augmentations during training
   train_transform = Compose([
       Resize((256, 256)),
       RandomHorizontalFlip(),
       RandomVerticalFlip(),
       ToTensor(),
       Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
   ])
   ```

## Creating Custom Pre-processors

### 1. Simple Transform Extension

```python
from torchvision.transforms.v2 import Transform

class CustomTransform(Transform):
    """Custom transform that scales pixel values."""

    def __init__(self, factor: float = 1.0):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        return img * self.factor

# Use in pre-processor
pre_processor = PreProcessor(
    transform=Compose([
        CustomTransform(factor=2.0),
        ToTensor()
    ])
)
```

### 2. Custom Pre-processor Class

```python
class CustomPreProcessor(PreProcessor):
    """Custom pre-processor with specialized behavior."""

    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        normalize: bool = True
    ):
        transforms = [Resize(image_size)]
        if normalize:
            transforms.extend([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
            ])

        super().__init__(transform=Compose(transforms))

    def forward(self, batch):
        """Add custom pre-processing logic."""
        processed = super().forward(batch)
        # Add custom processing here
        return processed
```

## Best Practices

1. **Transform Consistency**:

   - Use consistent image sizes across splits
   - Apply normalization consistently
   - Document transform requirements

2. **Performance**:

   - Use GPU-accelerated transforms when possible
   - Batch transforms for efficiency
   - Consider memory usage

3. **Validation**:

   - Verify transform output shapes
   - Check value ranges
   - Test with different input types

4. **Custom Transforms**:
   - Inherit from `Transform` base class
   - Implement forward/inverse methods
   - Handle edge cases

## Common Pitfalls

1. **Inconsistent Normalization**:

   - Different normalization between train/test
   - Missing normalization in custom transforms
   - Incorrect mean/std values

2. **Transform Order**:

   - Normalizing before augmentation
   - Converting to tensor too early
   - Missing crucial transforms

3. **Memory Issues**:
   - Large image sizes
   - Inefficient transforms
   - Memory leaks in custom transforms

```{seealso}
For more information about transforms:
- {doc}`Data Transforms Guide <../data/transforms>`
- {doc}`AnomalibModule Documentation <../../reference/models/base>`
```
