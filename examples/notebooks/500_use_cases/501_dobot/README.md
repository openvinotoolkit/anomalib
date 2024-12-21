# Dobot Use Case Tutorial

| Notebook                 | Link                                                                                                                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Training a Model         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501a_training_a_model_with_cubes_from_a_robotic_arm.ipynb) |
| Inference with the Model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501b_inference_with_a_robotic_arm.ipynb)                   |

## Steps to use these notebooks

These notebooks will help you to use a Dobot robot and Anomalib Library, for showcasing different kind of Industrial solutions

Step 1: Create an environment to run Anomalib + Dobot DLL using Python version 3.8

a. For Windows, use the following:

       python -m venv anomalib_env
       anomalib_env\Scripts\activate

b. For Linux and MacOS:

       python3 -m venv anomalib_env
       source anomalib_env/bin/activate

Step 2: Install Anomalib using pip install, you will find this command line in the notebook, but you can also install it after to run the notebook:

       pip install anomalib[full]

Step 3: Install Jupyter Lab or Jupyter Notebook through: https://jupyter.org/install

       pip install notebook
       pip install ipykernel
       pip install ipywidgets

## Notebook Contents

Step 4: Then connect your USB Camera and verify it works using a simple camera application. Once it is verified, close the application.

Step 5 (Optional): If you have the Dobot robot please make the following.

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

For data acquisition and inferencing we will use [501b notebook](501b_inference_with_a_robotic_arm.ipynb). There we need to identify the `acquisition` flag, **True** for _acquisition mode_ and **False** for _inferencing mode_. In acquisition mode be aware of the _normal_ or _abnormal_ folder we want to create, in this mode the notebook will save every image in the anomalib/datasets/cubes/{FOLDER} for further training. In inferencing mode the notebook won't save images, it will run the inference and show the results.

_Note_: If you dont have the robot you could jump to the another notebook [501a](501a_training_a_model_with_cubes_from_a_robotic_arm.ipynb) and download the dataset from this [link](https://github.com/openvinotoolkit/anomalib/releases/tag/dobot)

### Training

For training we will use [501a notebook](https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501a_training_a_model_with_cubes_from_a_robotic_arm.ipynb). In this example we are using "Padim" model and we are using Anomalib API for setting up the dataset, model, metrics, and the optimization process with OpenVINO.

### Have Fun and share your results in the discussion channel! 😊
