# RAD

This is the implementation of the [RAD](https://arxiv.org/abs/2601.22763) paper.

Model Type: Segmentation

## Description

RAD (Retrieval-based Anomaly Detection) is a training-free framework that detects anomalies through multi-level retrieval with position-aware patch matching. It uses a DINOv3 ViT-B/16 backbone to extract features from multiple transformer blocks and stores them in a multi-layer memory bank during the fitting phase.

At inference, the model first retrieves the k most similar training images using CLS token similarity, then performs patch-level anomaly scoring using cosine distance against a local memory built from the retrieved images. Position-aware matching restricts each query patch to compare only against patches from nearby spatial locations.

## Architecture

The model consists of three stages:

1. **Feature Extraction**: Multi-layer ViT features (CLS + patch tokens) from blocks 3, 6, 9, 11 (0-based)
2. **Image-level Retrieval**: Find k nearest training images via CLS token cosine similarity
3. **Patch-level Scoring**: Position-aware cosine distance between test patches and retrieved memory patches, fused across layers

## Usage

RAD is training-free: fitting only populates the memory bank, so a single pass over the anomaly-free training images is sufficient.

`anomalib train --model Rad --data MVTecAD --data.category <category>`

This builds one memory bank per category, which corresponds to the single-class setting of the paper (Sec. 5.4). The headline results are reported under the multi-class (MUAD) protocol, where a single bank is shared across every category of a dataset.

## Hyperparameters

The paper tunes the number of retrieved images and the neighborhood radius per dataset:

| Dataset  | `k_image` | `pos_radius` |
| -------- | --------- | ------------ |
| MVTec-AD | 150       | 1            |
| VisA     | 900       | 2            |
| Real-IAD | 900       | 0            |
| 3D-ADAM  | 48        | 1            |

Position-aware matching compares patches by grid location, so the fitting and inference image sizes must match. Use `use_positional_bank=False` if they differ.

## Memory Footprint

The memory bank stores every anomaly-free patch embedding and is saved inside the checkpoint, so both scale linearly with the size of the training set.

> **TODO**: Add architecture and sample-result images under `docs/source/images/rad/`, and benchmark numbers once measured artifacts are committed under `results/`.
