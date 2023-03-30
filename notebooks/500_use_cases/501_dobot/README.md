# Dobot Use Case Tutorial

| Notebook                       |                                                                                                                                                                                                                                                          |     |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| Dataset Creation and Inference | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501a_dataset_creation_and_Inference_with_a_robotic_arm.ipynb) |
| Training a Model               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501b_training_a_model_with_cubes_from_a_robotic_arm.ipynb)    |

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](https://openvinotoolkit.github.io/anomalib/getting_started/installation/index.html).

## Notebook Contents

This notebook demonstrates how NNCF can be used to compress a model trained with Anomalib. The notebook is divided into the following sections:

- Train an anomalib model without compression
- Train a model with NNCF compression
- Compare the performance of the two models (FP32 vs INT8)

Step 1: Then connect your USB Camera and verify it works using a simple camera application. Once it is verified, close the application.

Step 2 (Optional): If you have the Dobot robot please make the following.

1. Install Dobot requirements (See Dobot documentation here: https://en.dobot.cn/products/education/magician.html).
2. Check all connections to the Dobot and verify it is working using the Dobot Studio.
3. Install the vent accessory on the Dobot and verify it is working using Dobot Studio.
4. In the Dobot Studio, hit the "Home" button, and locate the:

![image](https://user-images.githubusercontent.com/10940214/219142393-c589f275-e01a-44bb-b499-65ebeb83a3dd.png)

a. Calibration coordinates: Initial position upper-left corner of cubes array.

b. Place coordinates: Position where the arm should leave the cubic over the conveyor belt.

c. Anomaly coordinates: Where you want to release the abnormal cube.

d. Then, replace those coordinates in the notebook

### Data acquisition and inferencing

For data acquisition and inferencing we will use [501_1 notebook](https://github.com/openvinotoolkit/anomalib/blob/feature/notebooks/usecases/dobot/notebooks/500_use_cases/dobot/501_1_Dataset%20creation%20and%20Inference%20with%20a%20robotic%20arm.ipynb). There we need to identify the `acquisition` flag, **True** for _acquisition mode_ and **False** for _inferencing mode_. In acquisition mode be aware of the _normal_ or _abnormal_ folder we want to create, in this mode the notebook will save every image in the anomalib/datasets/cubes/{FOLDER} for further training. In inferencing mode the notebook won't save images, it will run the inference and show the results.

_Note_: If you dont have the robot you could jump to the another notebook [501_2](https://github.com/openvinotoolkit/anomalib/blob/feature/notebooks/usecases/dobot/notebooks/500_use_cases/dobot/501_2_Training%20a%20model%20with%20cubes%20from%20a%20robotic%20arm.ipynb) and download the dataset from this [link](https://github.com/openvinotoolkit/anomalib/releases/tag/dobot)

### Training

For training we will use [501_2 notebook](https://github.com/openvinotoolkit/anomalib/blob/feature/notebooks/usecases/dobot/notebooks/500_use_cases/dobot/501_2_Training%20a%20model%20with%20cubes%20from%20a%20robotic%20arm.ipynb). In this example we are using "Padim" model and we are using Anomalib API for setting up the dataset, model, metrics, and the optimization process with OpenVINO.

**WIP**: Pending add some instructions here.

### Have Fun and share your results in the discussion channel! 😊
