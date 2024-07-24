"""Test metrics collection creation."""

from torchmetrics.classification import Accuracy

from anomalib.metrics import AUPRO, create_metric_collection


def test_string_initialization() -> None:
    """Pass metrics as a list of string."""
    metrics_list = ["AUROC", "AUPR"]
    collection = create_metric_collection(metrics_list, prefix=None)
    assert len(collection) == 2
    assert "AUROC" in collection
    assert "AUPR" in collection


def test_dict_initialization() -> None:
    """Pass metrics as a dictionary."""
    metrics_dict = {
        "PixelWiseAUROC": {
            "class_path": "anomalib.metrics.AUROC",
            "init_args": {},
        },
        "Precision": {
            "class_path": "torchmetrics.Precision",
            "init_args": {"task": "binary"},
        },
    }
    collection = create_metric_collection(metrics_dict, prefix=None)
    assert len(collection) == 2
    assert "PixelWiseAUROC" in collection
    assert "Precision" in collection


def test_metric_object_initialization() -> None:
    """Pass metrics as a list of metric objects."""
    metrics_list = [AUPRO(), Accuracy(task="binary")]
    collection = create_metric_collection(metrics_list, prefix=None)
    assert len(collection) == 2
    assert "AUPRO" in collection
    assert "BinaryAccuracy" in collection

    collection = create_metric_collection(AUPRO(), prefix=None)
    assert len(collection) == 1
    assert "AUPRO" in collection
