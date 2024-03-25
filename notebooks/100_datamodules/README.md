# Anomalib DataModules Tutorial

| Notebook | GitHub                         | Colab                                                                                                                                                                                                |
| -------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BTech    | [101_btech](101_btech.ipynb)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/100_datamodules/101_btech.ipynb)  |
| MVTec    | [102_mvtec](102_mvtec.ipynb)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/100_datamodules/102_mvtec.ipynb)  |
| Folder   | [103_folder](103_folder.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/100_datamodules/103_folder.ipynb) |
| Tiling   | [104_tiling](104_tiling.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/100_datamodules/104_tiling.ipynb) |

## Notebook Contents

The notebooks in this section demonstrate the mechanics of anomalib data modules, with a specific focus on benchmarks such as MVTec AD, BTech, and custom datasets via the Folder module. Anomalib data modules are structured as follows: Each data collection implements the Torch Dataset and the PyTorch Lightning DataModule objects.

The Torch Dataset inherits `torch.utils.data.Dataset` and implement the `__len__` and `__getitem__` methods. This implementation might therefore be utilized not just for anomalib, but also for other implementations.

The DataModule implementation inherits the PyTorch Lightning `DataModule` object. The advantage of this class is that it organizes each step of data from download to creating the Torch dataloader.

Overall, a data implementation has the following structure:

```bash
anomalib
├── __init__.py
├── data
│   ├── __init__.py
│   ├── btech.py
│   │   ├── BTechDataset
│   │   └── BTech
│   ├── folder.py
│   │   ├── FolderDataset
│   │   └── Folder
│   ├── inference.py
│   │   ├── InferenceDataset
│   │   mvtec.py
│   │   ├── MVTecDataset
└── └── └── MVTec
```

Let's deep dive into each dataset supported in anomalib and check their functionality.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](https://openvinotoolkit.github.io/anomalib/getting_started/installation/index.html).
