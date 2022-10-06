# CFA for Target-Oriented Anomaly Localization

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cfa-coupled-hypersphere-based-feature/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=cfa-coupled-hypersphere-based-feature)

PyTorch implementation of [CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization](https://arxiv.org/abs/2206.04325) (CFA).

## Getting Started

Install packages with:

```
$ pip install -r requirements.txt
```

## Dataset 

Prepare industrial image as:

``` 
train data:
    dataset_path/class_name/train/good/any_filename.png
    [...]

test data:
    dataset_path/class_name/test/good/any_filename.png
    [...]

    dataset_path/class_name/test/defect_type/any_filename.png
    [...]
``` 

## How to train

### Example
```
python trainer_cfa.py --class_name all --data_path [/path/to/dataset/] --cnn wrn50_2 --size 224 --gamma_c 1 --gamma_d 1
```

## Performance 
### WideResNet-50
R : resize. 
C : crop

|            |     R+C     |      R      |     CFA++
|------------|-------------|-------------|------------
| bottle     | 100  / 98.6 | 100  / 98.9 | 100  / 98.9
| cable      | 99.8 / 98.7 | 99.8 / 99.0 | 99.8 / 99.0
| capsule    | 97.3 / 98.9 | 99.2 / 99.1 | 99.2 / 99.1
| carpet     | 99.5 / 98.7 | 99.4 / 99.0 | 99.5 / 99.0 
| grid       | 99.2 / 97.8 | 99.9 / 98.1 | 99.9 / 98.1
| hazelnut   | 100  / 98.6 | 100  / 98.9 | 100  / 98.9
| leather    | 100  / 99.1 | 100  / 99.3 | 100  / 99.3
| metalnut   | 100  / 98.8 | 100  / 99.1 | 100  / 99.1
| pill       | 97.9 / 98.6 | 97.9 / 98.8 | 97.9 / 98.8
| screw      | 97.3 / 99.0 | 93.5 / 98.8 | 97.3 / 99.0
| tile       | 99.4 / 95.8 | 100  / 96.3 | 100  / 96.3
| toothbrush | 100  / 98.8 | 97.2 / 99.1 | 100  / 99.1
| transistor | 100  / 98.3 | 100  / 98.4 | 100  / 98.4
| wood       | 99.7 / 94.8 | 99.2 / 95.0 | 99.7 / 95.0
| zipper     | 99.6 / 98.6 | 99.5 / 99.0 | 99.6 / 99.0
| avg.       | 99.3 / 98.2 | 99.0 / 98.5 | 99.5 / 98.5


## Reference
[1] https://github.com/byungjae89/SPADE-pytorch

[2] https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

[3] https://github.com/pytorch/vision/tree/main/torchvision/models

[4] https://github.com/lukasruff/Deep-SVDD-PyTorch


## Citation

```
@article{lee2022cfa,
  title={CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization},
  author={Lee, Sungwook and Lee, Seunghyun and Song, Byung Cheol},
  journal={arXiv preprint arXiv:2206.04325},
  year={2022}
}
```
