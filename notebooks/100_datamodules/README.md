# Anomalib DataModules

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
