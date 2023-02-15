## Steps to use these notebooks


1. Create an environment to run Anomalib + Dobot DLL (Python version 3.8)
2. Install Anomalib. Follow the instructions here: https://github.com/openvinotoolkit/anomalib.
 ![image](https://user-images.githubusercontent.com/10940214/215845710-a7acaf6e-c4ed-4840-8bed-455674151a5d.png)

    a. Install Jupyter Lab or Jupyter Notebook: https://jupyter.org/install
    
    b. Install OpenVINO in your actual enviroment using this ```pip install -r requirements\openvino.txt``` 
3. Install Dobot requirements (See Dobot documentation here: https://en.dobot.cn/products/education/magician.html)
4. Check all connections of the Dobot, and verify if it is working using Dobot Studio.
5. Connect the WebCam and verify it works using a simple camera application. Close this app after the verification.
6. Install the vent on the dobot and verify if this is working using Dobot Studio.
7. In Dobot Studio, find "Home" by hitting the "Home" Button. Then, find the Calibration Coordinates (Initial position upper-left corner of cubes array), Place Coordinates (Position where the arm should leave the cubic over the conveyor belt), and Anomaly Coordinates (Where you want to release the abnormal cube. Replace those coordinates in the notebook. ![image](https://user-images.githubusercontent.com/10940214/198703796-3979d37d-ad9e-4e93-92b4-c575b1bde4b2.png)
Camera stop will be the same position that Place Coordinates with +90 in Z. See image below ![image](https://user-images.githubusercontent.com/10940214/198698536-9a1c403d-c7e3-4186-955b-4ceefb8fb379.png)
8. In the same environment where you have Dobot and Anomalib installed, verify you have jupyter notebooks, or Jupyter Lab installed.
9. Open [the main notebook](https://github.com/paularamo/cvpr-2022/blob/gh-pages/dobot/notebooks_control/Anomalib_Dobot_cubics_FINAL.ipynb) ![image](https://user-images.githubusercontent.com/10940214/198696689-1be3583d-0356-4305-a2cd-f51e4ff62409.png)
### Data acquisition
10. Verify the path folder of the dataset.
11. In cell #3, change the flag status to "True"![image](https://user-images.githubusercontent.com/10940214/198696596-459c97be-8789-4878-a038-1fa417a0b4c8.png)
12. Organize cubes with no abnormalities in the array, run the notebook, and verify that the notebook is creating the images.
13. Organize cubes with abnormalities in the array, rerun the notebook, and verify that the notebook is creating the images.
### Training
14. Save [cubes_config.yaml](https://github.com/paularamo/cvpr-2022/blob/gh-pages/dobot/cubes_config.yaml) the this path ```../../anomalib/models/{MODEL}/cubes_config.yaml```. See this link for understanding the config file creation, https://openvinotoolkit.github.io/anomalib/how_to_guides/train_custom_data.html.
15. Verify if this file has these two lines (See the highlights on the image below). If the answer is "Yes", please delete those three lines. ![image](https://user-images.githubusercontent.com/10940214/198704365-13b94a42-a9d9-4704-b9a5-6424c08fce9f.png)
16. Change the openvino_inference.py file to this one https://github.com/paularamo/cvpr-2022/blob/gh-pages/dobot/openvino_inference.py and modify this line of code with your path result. 
![image](https://user-images.githubusercontent.com/10940214/199284768-60ca5a53-aabc-4ba8-a293-db4bcf431f8a.png)

17. Before running [this notebook](
https://github.com/paularamo/cvpr-2022/blob/gh-pages/dobot/notebooks/001-getting-started-cubics/001-getting-started-Inference-cubics.ipynb), verify your dataset is well-connected with the notebook, verify that the inference is working and you can see the confidence result in the text file.
18. It will take some minutes to have the model ready. You don't need a GPU. If you have one, the training will be faster.
19. Verify where the model is saved. The model will be saved in the same folder of this notebook ``` ..\results\padim\cubes\weights\model.ckpt ```.
### Inference
20. Come back to [the main notebook](https://github.com/paularamo/cvpr-2022/blob/gh-pages/dobot/notebooks_control/Anomalib_Dobot_cubics_FINAL.ipynb).
21. Verify that you have the proper path for your created model. (Cell #2) ![image](https://user-images.githubusercontent.com/10940214/198702126-ee1c5e2b-a598-421a-98a3-743de5353028.png)
21. Verify that you have the inference text file linked properly. Same path as step #16.
22. In cell #3, change the flag status to "False"![image](https://user-images.githubusercontent.com/10940214/198696596-459c97be-8789-4878-a038-1fa417a0b4c8.png)
23. Run the notebook.
Have Fun! 😊
