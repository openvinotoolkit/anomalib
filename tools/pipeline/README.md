# Anomaly Detection in Industry Manufacturing

## Introduction

Anomalib is a deep learning library that aims to collect state-of-the-art anomaly detection algorithms for benchmarking on both public and private datasets. Anomalib provides several ready-to-use implementations of anomaly detection algorithms described in the recent literature, as well as a set of tools that facilitate the development and implementation of custom models. The library has a strong focus on image-based anomaly detection, where the goal of the algorithm is to identify anomalous images, or anomalous pixel regions within images in a dataset. Anomalib is constantly updated with new algorithms and training/inference extensions, so keep checking!

## Dataset
THE MVTEC ANOMALY DETECTION DATASET (MVTEC AD)

MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over 5000 high-resolution images divided into fifteen different object and texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as images without defects.

<img src='static/mvtec_dataset.jpg'>
Link dataset: <u>https://www.mvtec.com/company/research/datasets/mvtec-ad</u>

## Install packages
```
conda create -n anomaly-detection python=3.8
conda activate anomaly-detection
pip install -r requirements.txt
```
## Flow model
<img src='static/flow-app.jpg'>

## Custom dataset
For each new dataset, the data consist of three folders:
- train, which contains the (defect-free) training images
- test, which contains the test images
- ground_truth, which contains the pixel-precise annotations of anomalous regions
<img src='static/data_structure.jpg'>

Data custom sample: <a
            href="https://drive.google.com/drive/folders/1mUhEG-eHZetGabc9tC8YhvI8KIAu7AAn?usp=share_link">Link
            Drive</a>
## Train
```
python train.py --config "configs/patchcore.yaml" --model "patchcore"
```

## Evaluation (Coming soon)
## Inference
```
python script_inference.py --config "configs/patchcore_resnet18_aqa.yaml" --weight "models/patchcore_resnet18_aqa.ckpt" --image "static/aqa.png"
```
or just simple:
```
python script_inference.py
```
## App Flask
```
python app.py
```
Open local URL: http://127.0.0.1:5000

Default account login:
- Username: anomalib
- Password: anomalib

Login:
<img src='static/login_app.jpg'>

Signup:
<img src='static/signup_app.jpg'>

Homepage:
<img src='static/home_app.jpg'>

Train:
<img src='static/train_app.jpg'>

Inference:
<img src='static/inference_app.jpg'>

Database:
<img src='static/database_app.jpg'>

## Container
```
docker build -t anomaly:v1 .
docker run anomaly:v1
```

or just simple

```
docker-compose up
```
## Deploy AWS
First: Create EC2 instance 

Second: git clone and install related packages
```
git clone https://github.com/vnk8071/anomaly-detection-in-industry-manufacturing.git
```
Next: install Miniconda and Docker engine

- Miniconda: Follow link https://varhowto.com/install-miniconda-ubuntu-18-04/
- Docker engine: Follow link https://docs.docker.com/engine/install/ubuntu/

```
docker-compose up
```

Final: access link http://user-IPv4-public-ec2-aws
