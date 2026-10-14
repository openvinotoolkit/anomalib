# Anomalib Data Tutorial

| Notebook | GitHub                               | Colab                                                                                                                                                                                                             |
| -------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BTech    | [btech](datamodules/btech.ipynb)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-edge-platform/anomalib/blob/main/examples/notebooks/02_data/datamodules/btech.ipynb)   |
| MVTec    | [mvtec](datamodules/mvtec.ipynb)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-edge-platform/anomalib/blob/main/examples/notebooks/02_data/datamodules/mvtec.ipynb)   |
| Folder   | [folder](datamodules/folder.ipynb)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-edge-platform/anomalib/blob/main/examples/notebooks/02_data/datamodules/folder.ipynb)  |
| Tiling   | [tiling](utils/tiling.ipynb)         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-edge-platform/anomalib/blob/main/examples/notebooks/02_data/utils/tiling.ipynb)        |
| Tabular  | [tabular](datamodules/tabular.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-edge-platform/anomalib/blob/main/examples/notebooks/02_data/datamodules/tabular.ipynb) |
| Kaputt   | [kaputt](datamodules/kaputt.ipynb)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-edge-platform/anomalib/blob/main/examples/notebooks/02_data/datamodules/kaputt.ipynb)  |

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
│   │   ├── MVTecADDataset
└── └── └── MVTec
```

Let's deep dive into each dataset supported in anomalib and check their functionality.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html#installation).
