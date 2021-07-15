# PatchCore anomaly detection
Unofficial implementation of PatchCore(new SOTA) anomaly detection model


Original Paper : 
Towards Total Recall in Industrial Anomaly Detection (Jun 2021)  
Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler  


https://arxiv.org/abs/2106.08265  
https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad

![plot](./capture/capture.jpg)


updates(21/06/21) :  
- I used sklearn's SparseRandomProjection(ep=0.9) for random projection. I'm not confident with this.
- I think exact value of "b nearest patch-features" is not presented in the paper. I just set 9. (args.n_neighbors)  
- In terms of NN search, author used "faiss". but not implemented in this code yet. 
- sample embeddings/carpet/embedding.pickle => coreset_sampling_ratio=0.001  

updates(21/06/26) :  
- A critical [issue](https://github.com/hcw-00/PatchCore_anomaly_detection/issues/3#issue-930229038) related to "locally aware patch" raised and fixed. Score table is updated. 

### Usage 
~~~
# install python 3.6, torch==1.8.1, torchvision==0.9.1
pip install -r requirements.txt

python train.py --phase train or test --dataset_path .../mvtec_anomaly_detection --category carpet --project_root_path path/to/save/results --coreset_sampling_ratio 0.01 --n_neighbors 9'

# for fast try just specify your dataset_path and run
python train.py --phase test --dataset_path .../mvtec_anomaly_detection --project_root_path ./
~~~

### MVTecAD AUROC score (PatchCore-1%, mean of n trials)
| Category | Paper<br>(image-level) | This code<br>(image-level) | Paper<br>(pixel-level) | This code<br>(pixel-level) |
| :-----: | :-: | :-: | :-: | :-: |
| carpet | 0.980 | 0.991(1) | 0.989 | 0.989(1) |
| grid | 0.986 | 0.975(1) | 0.986 | 0.975(1) |
| leather | 1.000 | 1.000(1) | 0.993 | 0.991(1) |
| tile | 0.994 | 0.994(1) | 0.961 | 0.949(1) |
| wood | 0.992 | 0.989(1) | 0.951 | 0.936(1) |
| bottle | 1.000 | 1.000(1) | 0.985 | 0.981(1) |
| cable | 0.993 | 0.995(1) | 0.982 | 0.983(1) |
| capsule | 0.980 | 0.976(1) | 0.988 | 0.989(1) |
| hazelnut | 1.000 | 1.000(1) | 0.986 | 0.985(1) |
| metal nut | 0.997 | 0.999(1) | 0.984 | 0.984(1) |
| pill | 0.970 | 0.959(1) | 0.971 | 0.977(1) |
| screw | 0.964 | 0.949(1) | 0.992 | 0.977(1) |
| toothbrush | 1.000 | 1.000(1) | 0.985 | 0.986(1) |
| transistor | 0.999 | 1.000(1) | 0.949 | 0.972(1) |
| zipper | 0.992 | 0.995(1) | 0.988 | 0.984(1) |
| mean | 0.990 | 0.988 | 0.980 | 0.977 |

### Code Reference
kcenter algorithm :  
https://github.com/google/active-learning  
embedding concat function :  
https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

<!--
### MVTecAD AUROC score (PatchCore-1%, mean of n trials)
| Category | Paper<br>(image-level) | This code<br>(image-level) | Paper<br>(pixel-level) | This code<br>(pixel-level) |
| :-----: | :-: | :-: | :-: | :-: |
| carpet | 0.980 | 0.997(1) | 0.989 | 0.990(1) |
| grid | 0.986 | 0.941(1) | 0.986 | 0.983(1) |
| leather | 1.000 | 1.000(1) | 0.993 | 0.991(1) |
| tile | 0.994 | 0.982(1) | 0.961 | 0.932(1) |
| wood | 0.992 | 0.999(1) | 0.951 | 0.976(1) |
| bottle | 1.000 | 0.986(1) | 0.985 | 0.941(1) |
| cable | 0.993 | 0.970(1) | 0.982 | 0.955(1) |
| capsule | 0.980 | 0.949(1) | 0.988 | 0.987(1) |
| hazelnut | 1.000 | 0.997(1) | 0.986 | 0.982(1) |
| metal nut | 0.997 | 0.997(1) | 0.984 | 0.962(1) |
| pill | 0.970 | 0.918(1) | 0.971 | 0.941(1) |
| screw | 0.964 | 0.967(1) | 0.992 | 0.987(1) |
| toothbrush | 1.000 | 0.997(1) | 0.985 | 0.984(1) |
| transistor | 0.999 | 0.960(1) | 0.949 | 0.894(1) |
| zipper | 0.992 | 0.968(1) | 0.988 | 0.987(1) |
| mean | 0.990 | 0.975 | 0.980 | 0.966 |
-->
<!--
carpet
{'img_auc': 0.9907704654895666, 'pixel_auc': 0.988687705858456}
grid
{'img_auc': 0.974937343358396, 'pixel_auc': 0.9751171484318466}
leather
{'img_auc': 1.0, 'pixel_auc': 0.990535126453962}
tile
{'img_auc': 0.9935064935064934, 'pixel_auc': 0.9487486588602011}
wood
{'img_auc': 0.9894736842105263, 'pixel_auc': 0.9361264985782104}
bottle
{'img_auc': 1.0, 'pixel_auc': 0.9812954295492088}
cable
{'img_auc': 0.994752623688156, 'pixel_auc': 0.9827989792867079}
capsule
{'img_auc': 0.9764658954926206, 'pixel_auc': 0.9889956357045827}
hazelnut
{'img_auc': 1.0, 'pixel_auc': 0.9846421467788966}
matal nut
{'img_auc': 0.9990224828934506, 'pixel_auc': 0.9838413598506198}
pill
{'img_auc': 0.9593562465902891, 'pixel_auc': 0.9772725707677767}
screw
{'img_auc': 0.9491699118671859, 'pixel_auc': 0.9774047713027145}
toothbrush
{'img_auc': 1.0, 'pixel_auc': 0.9862281871506898}
transistor
{'img_auc': 0.9995833333333333, 'pixel_auc': 0.9715171862282481}
zipper
{'img_auc': 0.9952731092436975, 'pixel_auc': 0.9838086200703351}
>
