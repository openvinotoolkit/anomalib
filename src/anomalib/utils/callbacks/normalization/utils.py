import importlib

from lightning.pytorch import Callback
from omegaconf import DictConfig

from anomalib.post_processing import NormalizationMethod

from .cdf_normalization import _CdfNormalizationCallback
from .min_max_normalization import _MinMaxNormalizationCallback


def get_normalization_callback(
    normalization_method: NormalizationMethod | DictConfig | Callback | str = NormalizationMethod.NONE,
) -> Callback | None:
    """Return normalization object.

    normalization_method is an instance of ``Callback``, it is returned as is.

    if normalization_method is of type ``NormalizationMethod``, then a new class is created based on the type of
    normalization_method.

    Otherwise it expects a dictionary containing class_path and init_args.
        normalization_method:
            class_path: MinMaxNormalizer
            init_args:
                -
                -

    Example:
        >>> normalizer = get_normalization_callback(NormalizationMethod.MIN_MAX)
        or
        >>> normalizer = get_normalization_callback("min_max")
        or
        >>> normalizer = get_normalization_callback({"class_path": "MinMaxNormalizationCallback", "init_args": {}})
        or
        >>> normalizer = get_normalization_callback(MinMaxNormalizationCallback())
    """
    normalizer: Callback | None
    if isinstance(normalization_method, (NormalizationMethod, str)):
        normalizer = _get_normalizer_from_method(NormalizationMethod(normalization_method))
    elif isinstance(normalization_method, Callback):
        normalizer = normalization_method
    elif isinstance(normalization_method, DictConfig):
        normalizer = _parse_normalizer_config(normalization_method)
    else:
        raise ValueError(f"Unknown normalizer type {normalization_method}")
    return normalizer


def _get_normalizer_from_method(normalization_method) -> Callback | None:
    if normalization_method == NormalizationMethod.NONE:
        normalizer = None
    elif normalization_method == NormalizationMethod.MIN_MAX:
        normalizer = _MinMaxNormalizationCallback()
    elif normalization_method == NormalizationMethod.CDF:
        normalizer = _CdfNormalizationCallback()
    else:
        raise ValueError(f"Unknown normalization method {normalization_method}")
    return normalizer


def _parse_normalizer_config(normalization_method: DictConfig) -> Callback:
    class_path = normalization_method.class_path
    init_args = normalization_method.init_args

    if len(class_path.split(".")) == 1:
        module_path = "anomalib.utils.callbacks.normalization"
    else:
        module_path = ".".join(class_path.split(".")[:-1])
        class_path = class_path.split(".")[-1]
    module = importlib.import_module(module_path)
    class_ = getattr(module, class_path)
    normalizer = class_(**init_args)
    return normalizer
