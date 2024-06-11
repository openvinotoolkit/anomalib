# Region-Based Kernel Density Estimation (RKDE)

This is the implementation of the paper [Region Based Anomaly Detection With Real-Time
Training and Analysis](https://ieeexplore.ieee.org/abstract/document/8999287).

Model Type: Detection

## Description

Three-stage anomaly detection consisting of region extraction to obtain a set of region-of-interest proposals for each image, feature extraction to obtain a fixed-length feature vector for each region proposal, and density estimation to classify the region proposals as normal vs. anomalous.

Both the region extractor and the feature extractor rely on pre-trained convolutional neural networks. The density estimation stage uses Kernel Density Estimation (KDE).

### Region Extraction

Region proposals are obtained in the form of bounding boxes by feeding the images through a Faster-RCNN object detector with a ResNet50 backbone, pretrained on MS COCO. Depending on the chosen settings, the region proposals are obtained by taking either the final bounding box predictions of the classification heads, or the region proposals of the Region Proposal Network (RPN). Any detections with the `background` label are discarded, after which the raw region proposals are post-processed by discarding small bounding boxes, applying NMS (across all class labels), and discarding regions with a low confidence score. The minimum region size, IOU threshold used during NMS, and the confidence score threshold can be configured from the config file.

### Feature Extraction

The feature extractor consists of a Fast-RCNN model with an AlexNet backbone, which was trained in a multi-task setting on the MS COCO and Visual Genome datasets (see paper for more details). The ROI align layer ensures that the feature maps produced by the convolutional layers are cropped to the bounding box coordinates obtained in the region extraction stage. The activations of the final shared fully connected layer are retrieved to obtain a feature embeddings for each region proposal.

### Density Estimation

The classification module uses Kernel Density Estimation (KDE) to estimate the probability density function of the feature space. The KDE model is fitted on the collection of features extracted from the training images. During inference, features extracted from the regions in the inference images are evaluated against the KDE model to obtain a density estimation for each region proposal. The estimates density serves as a 'normality score', which is converted to a normal/anomalous label using Anomalib's thresholding mechanism.

Before fitting the KDE model, the dimensionality of the feature vectors is reduced using Principal Component Analysis (PCA). Depending on the chosen settings, the features are then scaled to unit vector length or the maximum vector length observed in the training set.

## Usage and parameters

`anomalib train --model Rkde --data MVTec --data.category <category>`

| Parameter                | Affects Stage      | Description                                                                                                                                                                                                | Type   | Options       |
| :----------------------- | :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----- | :------------ |
| roi_stage                | Region Extraction  | Processing stage from which the region proposals are retrieved. `rpn`: raw predictions of the region proposal network. `rcnn`: final detection outputs of the classification heads.                        | string | [rpn, rcnn]   |
| roi_score_threshold      | Region Extraction  | Minimum class score for the region proposals. Regions with a confidence score below this value are discarded. When stage is `rcnn`, class score is used. When stage is `rpn`, objectness score is used.    | float  |               |
| min_box_size             | Region Extraction  | Minimum size in pixels for the region proposals. Regions with a hight or width smaller than this value will be discarded.                                                                                  | int    |               |
| iou_threshold            | Region Extraction  | Intersection-Over-Union threshold used in Non-Maximum-Suppression when post-processing detections. Regions are discarded when their IoU with a higher-confidence region is above this value.               | float  |               |
| max_detections_per_image | Region Extraction  | Maximum number of region proposals N allowed per image. When the number of raw proposals is higher than this value, only the top N scoring proposals will be kept.                                         | int    |               |
| n_pca_components         | Density Estimation | Number of principal components to which the features are reduced before applying KDE.                                                                                                                      | int    |               |
| max_training_points      | Density Estimation | Maximum number of training features on which the KDE model is fitted. When more training features are available, a random selection of features will be discarded.                                         | int    |               |
| feature_scaling_method   | Density Estimation | Determines how the features are scaled before applying KDE. `norm`: the features are normalized to unit vector length. `scale`: The features are normalized to the max vector length observed in training. | string | [norm, scale] |

## Benchmark

N/A
