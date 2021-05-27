## Benchmark

### [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
### Image-Level AUC
|             |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| ----------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18   | 0.909 | 0.988  | 0.923 |  0.985  | 0.940 | 0.984 | 0.994  | 0.871 |  0.874  |  0.796   |   0.974   | 0.872 | 0.779 |   0.939    |   0.954    | 0.761  |
| Wide ResNet | 0.965 | 0.998  | 0.957 |  0.999  | 0.983 | 0.993 | 0.999  | 0.898 |  0.907  |          |   0.992   | 0.951 |       |   0.981    |   0.973    | 0.909  |


### Pixel-Level AUC
|             |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| ----------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18   | 0.964 | 0.986  | 0.919 |  0.992  | 0.916 | 0.937 | 0.980  | 0.957 |  0.980  |  0.972   |   0.957   | 0.951 | 0.973 |   0.986    |   0.968    | 0.980  |
| Wide ResNet | 0.974 | 0.990  | 0.970 |  0.991  | 0.940 | 0.954 | 0.982  | 0.963 |  0.985  |          |   0.974   | 0.961 |       |   0.988    |   0.973    | 0.986  |