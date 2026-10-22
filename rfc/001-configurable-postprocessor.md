# RFC: Consistent Pre/Post-Processing and Configurable Thresholds for Deploy Inferencers

- **Status:** Draft
- **Authors:** @ashwinvaidya17
- **Created:** 2026-04-29
- **Target Area:**
  - `src/anomalib/deploy/`
  - `src/anomalib/post_processing/post_processor.py`
  - `src/anomalib/models/components/base/export_mixin.py`
  - `src/anomalib/models/components/base/anomalib_module.py`

## Summary

This RFC proposes making anomaly detection sensitivity configurable at inference time by passing it as a graph input. `PostProcessor.forward()` gains `image_sensitivity` and `pixel_sensitivity` parameters. These become required inputs in the exported ONNX/OpenVINO graph alongside the image tensor. The graph always contains full pre- and post-processing — normalization and thresholding happen inside the graph, controlled by the sensitivity inputs.

An optional `metadata.json` sidecar captures pre-processing transforms (informational, for integrators) and default sensitivity values. Deploy inferencers (`TorchInferencer`, `OpenVINOInferencer`) read defaults from metadata and pass them to the graph, allowing operators to override sensitivity at construction or per-call without re-exporting.

## Motivation

`AnomalibModule.forward()` runs a three-stage pipeline during export:

```python
def forward(self, batch: torch.Tensor) -> InferenceBatch:
    batch = self.pre_processor(batch)
    batch = self.model(batch)
    return self.post_processor(batch)
```

Today, pre-processing and post-processing (including normalization and thresholding) are embedded directly in the exported model graph. This change was made so that we don't have to implement post-processing in the inference code for each backend. This reduces the bugs and maintenance overhead but it also means that:

- **Threshold/sensitivity cannot be changed at deploy time** without re-exporting the model. Tuning the operating point for a specific production environment requires a full re-export cycle.
- **Pre/post-processing logic is opaque** — downstream consumers cannot inspect what transforms, thresholds, or normalization stats the model uses.
- **No runtime configurability** — operators cannot adjust sensitivity (e.g., fewer false positives in low-risk areas, more sensitivity in critical areas) without going back to the training pipeline.

## Goals

1. **Configurable threshold/sensitivity**: Override at inference time without re-exporting.
2. **Transparent pre/post-processing**: Export metadata that documents the exact transforms, thresholds, and normalization stats baked into the model. This is useful for a) visualization where the model crops the image internally and we want to superimpose anomaly map on top b) Resize the image to the model size before transfering it to the device. This is also needed in inference under multi-processing scenarios as pickling a 4k image will slow down the pipeline. While transforms like cropping can still remain in the model so that it does not lead to unintended degradation.
3. **Runtime-adjustable operating point**: Let operators tune sensitivity per deployment environment via a single [0, 1] knob.

## Non-Goals

- Independent and isolated inference package. It will increase the scope and changes.
- Configurable pre-processor at inference time. The model graph already contains the full `pre_processor → model → post_processor` pipeline via `AnomalibModule.forward()`. The `preprocess` section in metadata is purely informational for integrators — not consumed by the inferencer.

## Design

### 1. Metadata Contract (`metadata.json`)

Exported alongside the model. Captures everything needed to replicate pre-processing and post-processing behavior.

```json
{
  "schema_version": "1.0",
  "anomalib_version": "2.1.0",
  "model": "Patchcore",
  "preprocess": [
    {
      "class_path": "torchvision.transforms.v2.Resize",
      "init_args": { "size": [256, 256], "antialias": false }
    },
    {
      "class_path": "anomalib.data.transforms.ExportableCenterCrop",
      "init_args": { "size": [224, 224] }
    },
    {
      "class_path": "torchvision.transforms.v2.Normalize",
      "init_args": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
      }
    }
  ],
  "postprocess": {
    "image_sensitivity": 0.5,
    "pixel_sensitivity": 0.5
  }
}
```

**`schema_version`**: The version of the metadata schema. Used by the migration system (Section 5) to apply upgrade/downgrade transformations when loading metadata produced by older or newer anomalib versions.

**`anomalib_version`**: The anomalib release that produced this metadata. Informational — the migration system uses `schema_version`, not this field, but it aids debugging.

**`model`**: The name of the model. This is used to identify the model in the metadata.

**`preprocess`**: `class_path` + `init_args` style (consistent with jsonargparse). **Informational only** — the model graph already contains the full pre-processing pipeline (see `AnomalibModule.forward()` which traces `pre_processor → model → post_processor` during export). This section exists so that integrators can inspect what transforms the model applies internally. Common uses: a) resize images to model input size _before_ transferring to the inference device (avoids pickling 4K images in multi-process pipelines), b) understand what crop was applied so anomaly maps can be superimposed on the original image, c) documentation. The inferencer does **not** reconstruct or apply transforms from this section.

**`postprocess`**: The default sensitivity values that were baked into the exported graph as input defaults. The inferencer reads these and passes them to the graph when the user does not provide an override. All other post-processing state (thresholds, min/max, enable flags) is embedded in the graph as `PostProcessor` buffers — not in metadata.

| metadata field      | Graph input                      |
| ------------------- | -------------------------------- |
| `image_sensitivity` | `image_sensitivity` input tensor |
| `pixel_sensitivity` | `pixel_sensitivity` input tensor |

### 2. Export: Sensitivity as Graph Inputs + Optional Metadata

The exported graph always contains the full pipeline (`pre_processor → model → post_processor`), with `image_sensitivity` and `pixel_sensitivity` as **required graph inputs** alongside the image tensor. Post-processing (normalization, thresholding) runs inside the graph, parameterized by the sensitivity inputs.

#### Graph input contract

| Input name          | Shape          | Type    | Description                    |
| ------------------- | -------------- | ------- | ------------------------------ |
| `input`             | `[B, 3, H, W]` | float32 | Image tensor                   |
| `image_sensitivity` | scalar         | float32 | Image-level sensitivity [0, 1] |
| `pixel_sensitivity` | scalar         | float32 | Pixel-level sensitivity [0, 1] |

#### AnomalibModule.forward() change

```python
def forward(self, batch: torch.Tensor,
            image_sensitivity: torch.Tensor,
            pixel_sensitivity: torch.Tensor) -> InferenceBatch:
    batch = self.pre_processor(batch) if self.pre_processor else batch
    batch = self.model(batch)
    if self.post_processor:
        return self.post_processor(batch, image_sensitivity, pixel_sensitivity)
    return batch
```

#### PostProcessor.forward() change

```python
def forward(self, predictions: InferenceBatch,
            image_sensitivity: torch.Tensor | None = None,
            pixel_sensitivity: torch.Tensor | None = None) -> InferenceBatch:
    img_sens = image_sensitivity if image_sensitivity is not None else self.image_sensitivity
    pix_sens = pixel_sensitivity if pixel_sensitivity is not None else self.pixel_sensitivity

    # normalization re-centered around effective threshold derived from sensitivity
    if self.enable_normalization:
        pred_score = self._normalize(pred_score, self.image_min, self.image_max,
                                     self._effective_threshold(self.image_threshold, self.image_min, self.image_max, img_sens))
        anomaly_map = self._normalize(predictions.anomaly_map, self.pixel_min, self.pixel_max,
                                      self._effective_threshold(self.pixel_threshold, self.pixel_min, self.pixel_max, pix_sens))

    # thresholding at fixed 0.5 on the re-centered normalized scores
    if self.enable_thresholding:
        pred_label = self._apply_threshold(pred_score, torch.tensor(0.5))
        pred_mask = self._apply_threshold(anomaly_map, torch.tensor(0.5))
    ...
```

During ONNX tracing, sensitivity is always a tensor (never `None`), so the traced graph always uses the parameter path. The `None` default is only for the Python API (training, callbacks).

#### Export implementation

```python
# In ExportMixin
def to_onnx(self, export_root, ..., write_metadata: bool = True) -> Path:
    ...
    default_img_sens = torch.tensor(self.post_processor.image_sensitivity)
    default_pix_sens = torch.tensor(self.post_processor.pixel_sensitivity)

    # Derive output names by running forward with sensitivity args.
    # Without passing sensitivity tensors here, forward() would still work
    # (falls back to module attributes), but we pass them explicitly to
    # match the exact tracing path used by torch.onnx.export below.
    output_names = [
        name for name, value in
        self.eval()(input_shape, default_img_sens, default_pix_sens)._asdict().items()
        if value is not None
    ]

    torch.onnx.export(
        model=self,
        args=(input_shape, default_img_sens, default_pix_sens),
        input_names=["input", "image_sensitivity", "pixel_sensitivity"],
        output_names=output_names,
        ...
    )
    if write_metadata:
        self._write_metadata(export_root)
    return onnx_path

def _write_metadata(self, export_root: Path) -> Path:
    import anomalib
    from anomalib.deploy.metadata import CURRENT_SCHEMA_VERSION

    metadata = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "anomalib_version": anomalib.__version__,
        "model": self.__class__.__name__,
        "preprocess": self._serialize_preprocess(self.pre_processor),
        "postprocess": {
            "image_sensitivity": self.post_processor.image_sensitivity,
            "pixel_sensitivity": self.post_processor.pixel_sensitivity,
        },
    }
    path = export_root / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2))
    return path
```

When `metadata.json` is absent, the inferencer uses a default sensitivity of 0.5 (threshold is centered around 0.5 so using that as the default value is reasonable). When present, the inferencer reads the trained defaults from metadata and passes them to the graph.

### 3. Inference: Pass Sensitivity to Graph

The inferencer does **no post-processing math**. It reads default sensitivity from `metadata.json` (or uses 0.5 if absent), applies any user overrides, and passes the values to the graph as input tensors. The graph handles everything.

**Pre-processing** is handled by the model graph itself. `AnomalibModule.forward()` traces `pre_processor → model → post_processor` during ONNX/OpenVINO export, so the exported graph already contains the input transforms. The `preprocess` section in metadata is not consumed by the inferencer — it is documentation for integrators (see Section 1).

**Post-processing** runs entirely inside the graph via `PostProcessor.forward()`. The inferencer's only job is to supply the sensitivity inputs:

```python
class OpenVINOInferencer:
    def __init__(self, path, device="AUTO", config=None,
                 image_sensitivity=None, pixel_sensitivity=None):
        ...
        self.metadata = self._load_metadata(path)
        self._default_image_sensitivity = (
            image_sensitivity
            or self.metadata.get("postprocess", {}).get("image_sensitivity", 0.5)
        )
        self._default_pixel_sensitivity = (
            pixel_sensitivity
            or self.metadata.get("postprocess", {}).get("pixel_sensitivity", 0.5)
        )

    def predict(self, image, *, image_sensitivity=None, pixel_sensitivity=None):
        image = self.pre_process(image)
        img_sens = image_sensitivity if image_sensitivity is not None else self._default_image_sensitivity
        pix_sens = pixel_sensitivity if pixel_sensitivity is not None else self._default_pixel_sensitivity

        predictions = self.model({
            self.input_blob.any_name: image,
            "image_sensitivity": np.array(img_sens, dtype=np.float32),
            "pixel_sensitivity": np.array(pix_sens, dtype=np.float32),
        })
        return NumpyImageBatch(image=image, **self.post_process(predictions))
```

**Backward compatibility with old exports:** Old models have a single input (`input`). The inferencer detects this by checking the model's input count and skips passing sensitivity — the graph uses its baked-in post-processing as before.

### 4. Runtime Sensitivity Override

#### How It Works

Sensitivity is a [0, 1] knob that controls the normalization center inside the graph. Instead of shifting the binary decision boundary, sensitivity computes an effective threshold that re-centers the normalization itself. Both the reported anomaly score and the binary label are affected:

```python
# Inside PostProcessor.forward() (in the graph)
eff_threshold = learned_threshold + (max - min) * (0.5 - sensitivity)
normalized = ((raw_score - eff_threshold) / (max - min)) + 0.5   # clamp [0, 1]
label = normalized > 0.5
```

At `sensitivity=0.5`, the effective threshold equals the learned threshold (no change — same as current behavior). Higher sensitivity lowers the effective threshold (scores shift up, more detections). Lower sensitivity raises it (scores shift down, fewer detections). The binary cutoff is always 0.5 — sensitivity controls what "0.5" means in the score space.

#### `_apply_threshold` change

Today, `_apply_threshold` receives the threshold from a property that reads `self.image_sensitivity` — a module attribute baked into the graph at trace time. The change makes normalization use an effective threshold derived from the sensitivity argument, while the binary cutoff becomes a fixed 0.5:

```diff
      def forward(self, predictions: InferenceBatch,
 +                image_sensitivity: torch.Tensor | None = None,
 +                pixel_sensitivity: torch.Tensor | None = None) -> InferenceBatch:
 +        img_sens = image_sensitivity if image_sensitivity is not None else self.image_sensitivity
 +        pix_sens = pixel_sensitivity if pixel_sensitivity is not None else self.pixel_sensitivity
          ...
 +        # sensitivity shifts normalization center, not the decision boundary
          if self.enable_normalization:
 -            pred_score = self._normalize(pred_score, self.image_min, self.image_max, self.image_threshold)
 -            anomaly_map = self._normalize(anomaly_map, self.pixel_min, self.pixel_max, self.pixel_threshold)
 +            pred_score = self._normalize(pred_score, self.image_min, self.image_max,
 +                                         self._effective_threshold(self.image_threshold, self.image_min, self.image_max, img_sens))
 +            anomaly_map = self._normalize(anomaly_map, self.pixel_min, self.pixel_max,
 +                                          self._effective_threshold(self.pixel_threshold, self.pixel_min, self.pixel_max, pix_sens))
          if self.enable_thresholding:
 -            pred_label = self._apply_threshold(pred_score, self.normalized_image_threshold)
 -            pred_mask = self._apply_threshold(anomaly_map, self.normalized_pixel_threshold)
 +            pred_label = self._apply_threshold(pred_score, torch.tensor(0.5))
 +            pred_mask = self._apply_threshold(anomaly_map, torch.tensor(0.5))
```

The new `_effective_threshold` helper converts sensitivity to a shifted threshold in the raw score space:

```python
@staticmethod
def _effective_threshold(
    threshold: torch.Tensor,
    norm_min: torch.Tensor,
    norm_max: torch.Tensor,
    sensitivity: torch.Tensor,
) -> torch.Tensor:
    return threshold + (norm_max - norm_min) * (0.5 - sensitivity)
```

```python
# _apply_threshold (unchanged)
@staticmethod
def _apply_threshold(preds, threshold):
    if preds is None or threshold.isnan():
        return preds
    return preds > threshold
```

The `normalized_image_threshold` and `normalized_pixel_threshold` properties become unused in the traced graph path and can be deprecated.

#### `_apply_threshold` internals

Current implementation:

```python
@staticmethod
def _apply_threshold(
    preds: torch.Tensor | None,
    threshold: torch.Tensor,
) -> torch.Tensor | None:
    if preds is None or threshold.isnan():
        return preds
    return preds > threshold
```

With this new design, `_apply_threshold` is always called with a fixed `torch.tensor(0.5)`, so the `isnan()` guard is never needed. The sensitivity logic lives entirely in `_effective_threshold` + `_normalize`. During ONNX tracing, the `preds is None` branch is resolved statically — the tracer takes one path and bakes it into the graph.

```diff
 @staticmethod
 def _apply_threshold(
     preds: torch.Tensor | None,
     threshold: torch.Tensor,
 ) -> torch.Tensor | None:
-    if preds is None or threshold.isnan():
+    if preds is None:
         return preds
     return preds > threshold
```

The `isnan()` guard can be removed — the threshold is always `0.5` in the traced path. The `preds is None` check remains for Python-side usage (e.g., classification-only models with no anomaly map) but is resolved statically during tracing.

#### Override Precedence

**per-call argument > constructor argument > metadata.json value > 0.5 fallback**

The inferencer resolves the effective sensitivity and passes it to the graph as a scalar tensor. The graph does all the math — the inferencer has zero post-processing logic.

#### Why Sensitivity Instead of Raw Threshold

Exposing the raw threshold directly was considered (user provides a value in the original score space). Rejected because:

- Raw scores are model-dependent, dataset-dependent, and training-run-dependent.
- A value like `50.0` is meaningless without context.
- Not portable across models.

Sensitivity is portable: `0.7` means "somewhat more sensitive than optimal" regardless of model. `0.5` always means "use the trained default." Power users who need exact threshold control can inspect the `PostProcessor` buffers in the graph directly.

### 5. Metadata Versioning & Migration

The `schema_version` field in `metadata.json` drives a lightweight migration system. Both writer and reader live in the same repository and schema versions will change infrequently, so the system is deliberately simple: a chain of pure upgrade functions, no registry, no decorators, no downgrade support (added only if a concrete need arises).

#### Module Layout

```
src/anomalib/deploy/metadata/
├── __init__.py          # re-exports load_metadata, dump_metadata, CURRENT_SCHEMA_VERSION
├── migration.py         # upgrade_to_latest() + per-version upgrade functions
└── schema.py            # dataclass/TypedDict schemas per version (validation)
```

#### Migration Implementation

```python
# src/anomalib/deploy/metadata/migration.py
from __future__ import annotations

import json
from pathlib import Path

CURRENT_SCHEMA_VERSION = "1.0"


def upgrade_to_latest(metadata: dict) -> dict:
    """Upgrade metadata dict to the current schema version.

    Applies each step in sequence. Pure function — does not mutate input.
    """
    metadata = metadata.copy()
    version = metadata.get("schema_version")

    if version is None:
        metadata = _legacy_to_v1(metadata)

    # Future upgrades go here:
    # if metadata["schema_version"] < "1.1":
    #     metadata = _v1_to_v1_1(metadata)

    return metadata


def _legacy_to_v1(raw: dict) -> dict:
    """Convert OmegaConf-style legacy metadata to schema 1.0."""
    return {
        "schema_version": "1.0",
        "anomalib_version": "unknown",
        "model": raw.get("model", "unknown"),
        "preprocess": [],  # legacy metadata did not capture transforms
        "postprocess": {
            "image_sensitivity": 0.5,
            "pixel_sensitivity": 0.5,
        },
    }
```

#### Load / Dump Helpers

```python
# src/anomalib/deploy/metadata/__init__.py
from anomalib.deploy.metadata.migration import CURRENT_SCHEMA_VERSION, upgrade_to_latest


def load_metadata(path: Path) -> dict:
    """Read metadata.json, auto-upgrade to current schema, validate."""
    raw = json.loads(path.read_text())
    metadata = upgrade_to_latest(raw)
    _validate(metadata)
    return metadata


def dump_metadata(metadata: dict, path: Path) -> None:
    """Validate and write metadata.json."""
    _validate(metadata)
    path.write_text(json.dumps(metadata, indent=2))


def _validate(metadata: dict) -> None:
    """Validate metadata against expected schema fields.

    Raises SchemaValidationError on mismatch.
    """
    required = {"schema_version", "anomalib_version", "model", "postprocess"}
    missing = required - metadata.keys()
    if missing:
        msg = f"Metadata missing required fields: {missing}"
        raise SchemaValidationError(msg)
```

#### How Inferencers Use It

```python
class OpenVINOInferencer:
    def __init__(self, path, ...):
        ...
        metadata_path = self._find_metadata(path)
        if metadata_path is not None:
            self.metadata = load_metadata(metadata_path)
        else:
            self.metadata = None  # fallback to legacy behavior
```

The inferencer never interprets `schema_version` directly — `upgrade_to_latest()` handles all translation before the inferencer sees the dict. Downgrade support is intentionally omitted; it will be added only when a concrete deployment scenario requires it.

### 6. Inferencer Version Support & Backwards Compatibility

Inferencers must gracefully handle metadata produced by older (and cautiously, newer) anomalib versions. This section defines the compatibility contract.

#### Version Fields in Metadata

| Field              | Purpose                                                              |
| ------------------ | -------------------------------------------------------------------- |
| `schema_version`   | Drives migration. Determines which fields exist and their semantics. |
| `anomalib_version` | Informational. The anomalib release that exported the model.         |

When loading metadata:

1. **No `schema_version` field**: Treat as legacy/pre-schema metadata. `upgrade_to_latest()` applies `_legacy_to_v1()` automatically.
2. **`schema_version` is current**: Load directly.
3. **`schema_version` is older**: `upgrade_to_latest()` applies each step in sequence.
4. **`schema_version` is newer** (from a future anomalib): Issue a warning, attempt best-effort load (ignore unknown keys, use defaults for missing ones). If critical fields are missing, raise `MetadataVersionError`.

```python
def _load_and_migrate(self, metadata_path: Path | None) -> dict | None:
    if metadata_path is None or not metadata_path.exists():
        return None

    metadata = load_metadata(metadata_path)  # auto-upgrades + validates

    if metadata["schema_version"] > CURRENT_SCHEMA_VERSION:
        logger.warning(
            "Metadata schema %s is newer than this anomalib supports (%s). "
            "Unknown fields will be ignored. Consider upgrading anomalib.",
            metadata["schema_version"], CURRENT_SCHEMA_VERSION,
        )

    return metadata
```

#### Serialization Round-Trip Guarantee

For any supported schema version `V`:

```
load_metadata(dump_metadata(metadata, path)) == metadata
```

This is enforced by golden-file tests: each schema version ships a `tests/fixtures/metadata_v{X}.json` that is loaded, upgraded, and diffed.

#### Deprecation Policy

| Schema version age    | Behavior                                           |
| --------------------- | -------------------------------------------------- |
| Current and current-1 | Fully supported, silent migration                  |
| Current-2             | Supported with deprecation warning at load time    |
| Older                 | `MetadataVersionError` with re-export instructions |

This gives users roughly 2 minor releases to re-export models before old metadata becomes unsupported.

## API Diffs

### PostProcessor (core change)

```diff
 class PostProcessor(nn.Module):
-    def forward(self, predictions: InferenceBatch) -> InferenceBatch:
+    def forward(self, predictions: InferenceBatch,
+                image_sensitivity: torch.Tensor | None = None,
+                pixel_sensitivity: torch.Tensor | None = None) -> InferenceBatch:
+        img_sens = image_sensitivity if image_sensitivity is not None else self.image_sensitivity
+        pix_sens = pixel_sensitivity if pixel_sensitivity is not None else self.pixel_sensitivity
         ...
+        if self.enable_normalization:
-            pred_score = self._normalize(pred_score, self.image_min, self.image_max, self.image_threshold)
-            anomaly_map = self._normalize(anomaly_map, self.pixel_min, self.pixel_max, self.pixel_threshold)
+            pred_score = self._normalize(pred_score, self.image_min, self.image_max,
+                                         self._effective_threshold(self.image_threshold, self.image_min, self.image_max, img_sens))
+            anomaly_map = self._normalize(anomaly_map, self.pixel_min, self.pixel_max,
+                                          self._effective_threshold(self.pixel_threshold, self.pixel_min, self.pixel_max, pix_sens))
         if self.enable_thresholding:
-            pred_label = self._apply_threshold(pred_score, self.normalized_image_threshold)
-            pred_mask = self._apply_threshold(anomaly_map, self.normalized_pixel_threshold)
+            pred_label = self._apply_threshold(pred_score, torch.tensor(0.5))
+            pred_mask = self._apply_threshold(anomaly_map, torch.tensor(0.5))
```

### AnomalibModule (graph signature change)

```diff
 class AnomalibModule:
-    def forward(self, batch):
+    def forward(self, batch, image_sensitivity=None, pixel_sensitivity=None):
         batch = self.pre_processor(batch) if self.pre_processor else batch
         batch = self.model(batch)
-        return self.post_processor(batch) if self.post_processor else batch
+        if self.post_processor:
+            return self.post_processor(batch, image_sensitivity, pixel_sensitivity)
+        return batch
```

### OpenVINOInferencer (passes sensitivity to graph)

```diff
 class OpenVINOInferencer:
-    def __init__(self, path, device="AUTO", config=None):
+    def __init__(self, path, device="AUTO", config=None,
+                 image_sensitivity=None, pixel_sensitivity=None):
         ...
+        self.metadata = self._load_metadata(path)

-    def predict(self, image):
+    def predict(self, image, *, image_sensitivity=None, pixel_sensitivity=None):
         image = self.pre_process(image)
-        predictions = self.model({self.input_blob.any_name: image})
+        predictions = self.model({
+            self.input_blob.any_name: image,
+            "image_sensitivity": np.array(img_sens, dtype=np.float32),
+            "pixel_sensitivity": np.array(pix_sens, dtype=np.float32),
+        })
         pred_dict = self.post_process(predictions)
         return NumpyImageBatch(image=image, **pred_dict)
```

### TorchInferencer (passes sensitivity as forward args)

```diff
 class TorchInferencer:
-    def __init__(self, path, device="auto"):
+    def __init__(self, path, device="auto",
+                 image_sensitivity=None, pixel_sensitivity=None):
         ...
+        self.metadata = self._load_metadata(path)

-    def predict(self, image):
+    def predict(self, image, *, image_sensitivity=None, pixel_sensitivity=None):
         image = self.pre_process(image)
-        predictions = self.model(image)
+        predictions = self.model(image,
+            torch.tensor(img_sens), torch.tensor(pix_sens))
         return ImageBatch(image=image, **predictions._asdict())
```

### Export Methods

```diff
 class ExportMixin:
     def to_onnx(self, export_root, ..., write_metadata: bool = True) -> Path:
         ...
-        torch.onnx.export(model=self, args=(input_shape,),
-                          input_names=["input"], ...)
+        torch.onnx.export(model=self,
+                          args=(input_shape, default_img_sens, default_pix_sens),
+                          input_names=["input", "image_sensitivity", "pixel_sensitivity"], ...)
+        if write_metadata:
+            self._write_metadata(export_root)
         return onnx_path
```

## Usage Examples

### Default (consistent across runtimes)

```python
from anomalib.deploy import OpenVINOInferencer

inferencer = OpenVINOInferencer(path="weights/model.xml")
pred = inferencer.predict("sample.png")
# pred.pred_score is normalized [0, 1], pred.pred_label is thresholded at 0.5
# Sensitivity defaults read from metadata.json (or 0.5 if absent)
```

### More sensitive (catch more anomalies)

```python
inferencer = OpenVINOInferencer(
    path="weights/model.xml",
    image_sensitivity=0.7,  # higher = more sensitive (lowers threshold)
)
```

### Less sensitive (fewer false positives)

```python
inferencer = OpenVINOInferencer(
    path="weights/model.xml",
    image_sensitivity=0.3,  # lower = less sensitive (raises threshold)
)
```

### Override sensitivity per-call

```python
pred = inferencer.predict("sample.png", image_sensitivity=0.9)
```

### Direct graph usage (no inferencer)

```python
import openvino as ov
compiled = ov.Core().compile_model(ov.Core().read_model("model.xml"), "CPU")
result = compiled({
    "input": image,
    "image_sensitivity": np.array(0.7, dtype=np.float32),
    "pixel_sensitivity": np.array(0.5, dtype=np.float32),
})
```

## Migration

### Backward Compatibility

- **New exports** have 3 graph inputs (`input`, `image_sensitivity`, `pixel_sensitivity`). All are required. The inferencer always provides sensitivity values.
- **Old exports** (pre-change, 1 input) continue to work. The inferencer detects the input count and omits sensitivity — the graph uses its baked-in post-processing as before.
- **Python API**: `AnomalibModule.forward(batch)` still works during training because sensitivity defaults to `None`, falling back to `PostProcessor.image_sensitivity`. The `None` path is only used in non-traced code (training, callbacks).
- **`write_metadata`** defaults to `True`. Metadata absence is the footgun (silent fallback to 0.5), not presence. Users who want lean artifacts can opt out with `write_metadata=False`.
- No API breakage — all new parameters are optional with `None`/`False` defaults.

### Rollout

1. **PostProcessor change**: Add `image_sensitivity`, `pixel_sensitivity` parameters to `PostProcessor.forward()`. Add `_effective_threshold()` with sensitivity clamping. Deprecate `normalized_image_threshold` / `normalized_pixel_threshold` properties.
2. **AnomalibModule change**: Add sensitivity parameters to `AnomalibModule.forward()`, pass through to `PostProcessor`.
3. **Export**: Update `to_onnx()` to trace with 3 inputs — including `output_names` derivation. Add `_write_metadata()` helper. Add `write_metadata: bool = True` parameter to `to_onnx`, `to_openvino`, `to_torch`.
4. **Metadata infrastructure**: Create `src/anomalib/deploy/metadata/` with `load_metadata`, `dump_metadata`, `upgrade_to_latest()`, and schema validation.
5. **Legacy migration**: Implement `_legacy_to_v1()` inside `migration.py` to bridge the existing OmegaConf-based format.
6. **OpenVINOInferencer**: Add sensitivity parameters to constructor and `predict()`. Detect graph input count for backward compat. Pass sensitivity tensors to graph.
7. **TorchInferencer**: Add sensitivity parameters to constructor and `predict()`. Pass as `forward()` args.
8. **Golden tests**: Export model → load with each inferencer → assert outputs match `Engine.predict()`. Test sensitivity override changes labels. Add migration round-trip tests with fixture files per schema version. Add training/export path consistency test (see Testing section).

## Risks

- **Graph input contract change**: New exports require 3 inputs. Direct graph users (outside our inferencers) who hardcoded a single input will need to update their code. Mitigated by metadata documentation and clear error messages from the runtime.
- **Metadata staleness**: If model is re-trained but metadata is not re-exported, default sensitivity may not match the trained values. Mitigated by encouraging `write_metadata=True` in export workflows.
- **Preprocess section trust**: The `preprocess` array uses `class_path` strings. Since it is informational-only (not imported or executed by the inferencer), there is no code-execution risk. However, integrators who choose to reconstruct the pipeline from metadata in their own code should use an allowlist.

## Testing

- Golden tests: export model → load with each inferencer → assert outputs match `Engine.predict()` output.
- Override tests: verify sensitivity override changes predictions as expected.
- Missing metadata tests: verify fallback to current behavior.
- Cross-runtime parity: same model, same image → `TorchInferencer` and `OpenVINOInferencer` produce identical normalized scores (within floating-point tolerance).
- **Training/export path consistency**: `forward()` (used by export tracing) and `post_process_batch()` (used by Lightning training hooks) must produce identical results at default sensitivity. Add a parametrized test that runs both paths on the same input batch with several sensitivity values and asserts outputs match within floating-point tolerance. This catches any drift between the two code paths as the codebase evolves.
- Metadata migration round-trip: golden-file tests per schema version — load fixture, upgrade, validate, diff.

## Appendix: JSON over YAML

JSON for deploy artifacts. Deterministic, schema-first, cross-language. YAML acceptable for authoring config only.

---

### Remaining Concerns

**A. Pixel-level sensitivity for classification-only models.** The `pixel_sensitivity` graph input exists even when the model produces no anomaly map. The `PostProcessor` already handles this (thresholding is skipped when `anomaly_map` is `None`), but the exported graph will always have the input node. The inferencer always passes `pixel_sensitivity` — it is a no-op for classification-only models. Document this in the inferencer docstring and metadata spec.

**B. Old exports (1 input) vs new exports (3 inputs).** The inferencer must detect whether a loaded model has sensitivity inputs. Proposed approach: check `len(compiled.inputs)` or inspect input names. This is straightforward but should be tested explicitly with old exported models.

**C. Sensitivity clamping.** `_effective_threshold` should clamp the sensitivity input to `[0, 1]` inside the graph to prevent nonsensical thresholds from out-of-range values:

```python
@staticmethod
def _effective_threshold(
    threshold: torch.Tensor,
    norm_min: torch.Tensor,
    norm_max: torch.Tensor,
    sensitivity: torch.Tensor,
) -> torch.Tensor:
    sensitivity = sensitivity.clamp(0.0, 1.0)
    return threshold + (norm_max - norm_min) * (0.5 - sensitivity)
```

**D. Sensitivity batch semantics.** Sensitivity inputs are scalar (`shape=()` or `(1,)`) — one value per inference call, applied uniformly to the batch. Per-image sensitivity (`shape=(B,)`) is not supported in this RFC. If needed later, it can be added as a backward-compatible extension (broadcast rules already work).

**E. Deprecation of `normalized_image_threshold` / `normalized_pixel_threshold`.** These properties become dead code in the traced graph path after this change. They will be deprecated with a warning in the same release and removed in the next minor version. The `threshold_batch()` and `normalize_batch()` training-time helpers continue to use them until the training path is unified (see Testing section for the consistency test that guards against drift).

**F. `write_metadata` default.** Changed from `False` to `True` in the export methods. Metadata absence is the footgun (inferencer silently falls back to 0.5), not metadata presence. Users who want lean artifacts can opt out with `write_metadata=False`.

## Appendix: Reproducer — Sensitivity as Graph Input

A standalone reproducer validates that sensitivity can be passed as an ONNX/OpenVINO graph input and overridden at inference time.

### What the reproducer tests

A `MiniPostProcessor` accepts `image_sensitivity` as a `forward()` argument. It computes an effective threshold from sensitivity, normalizes scores using baked-in buffers (threshold, min, max) centered around the effective threshold, and applies a fixed 0.5 cutoff. A `MiniModel` wraps a backbone + post-processor, exported to ONNX with 2 inputs: `input` (image) and `image_sensitivity` (scalar).

### Results

| Test | Runtime      | Sensitivity                     | Score   | Label   | Status                              |
| ---- | ------------ | ------------------------------- | ------- | ------- | ----------------------------------- |
| 1    | ONNX Runtime | 0.5 (explicit)                  | 0.7022  | 1       | Matches PyTorch                     |
| 2    | ONNX Runtime | 0.9 (override)                  | 1.0000  | 1       | Score shifts up, more detections    |
| 3    | ONNX Runtime | 0.1 (override)                  | 0.3022  | 0       | Score shifts down, fewer detections |
| 4    | ONNX Runtime | omitted (initializer default)   | 0.7022  | 1       | Initializer works                   |
| 5a   | OpenVINO     | ONNX w/ initializer             | 0.7021  | 1       | Default OK, but **not overridable** |
| 6    | OpenVINO     | required input, 0.5 / 0.9 / 0.1 | correct | correct | Override works                      |

### Key finding: OpenVINO folds ONNX initializers

When an ONNX input has a matching initializer (the standard pattern for optional inputs), OpenVINO folds it into a graph constant. The input disappears from the model's input list and cannot be overridden at runtime. ONNX Runtime handles it correctly (initializer serves as default, input overrides it).

**Decision: No initializers. Always-required inputs.** The ONNX/OpenVINO graph has `input`, `image_sensitivity`, and `pixel_sensitivity` as required inputs. The inferencer always provides sensitivity values — reading the default from `metadata.json` when the user doesn't override. This works identically across ONNX Runtime and OpenVINO.

### Reproducer code (simplified core)

```python
class MiniPostProcessor(nn.Module):
    def __init__(self, image_sensitivity: float = 0.5) -> None:
        super().__init__()
        self.register_buffer("_image_threshold", torch.tensor(0.42))
        self.register_buffer("image_min", torch.tensor(0.02))
        self.register_buffer("image_max", torch.tensor(0.91))

    def forward(self, pred_score, image_sensitivity):
        eff_threshold = self._image_threshold + (self.image_max - self.image_min) * (0.5 - image_sensitivity)
        normalized = ((pred_score - eff_threshold)
                      / (self.image_max - self.image_min)) + 0.5
        normalized = normalized.clamp(0, 1)
        pred_label = (normalized > 0.5).float()
        return normalized, pred_label

# Export with 2 inputs
torch.onnx.export(model, (input_tensor, torch.tensor(0.5)),
                  "model.onnx",
                  input_names=["input", "image_sensitivity"], ...)

# OpenVINO inference with override
compiled = ov.Core().compile_model(ov.Core().read_model("model.onnx"), "CPU")
result = compiled({"input": image, "image_sensitivity": np.array(0.9, dtype=np.float32)})
```

Full reproducer in the PR description.
