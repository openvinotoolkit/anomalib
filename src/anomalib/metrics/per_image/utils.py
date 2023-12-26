"""Torch-oriented interfaces for `utils.py`."""
from torch import Tensor

from . import _validate, utils_numpy
from .utils_numpy import StatsOutliersPolicy, StatsRepeatedPolicy


def per_image_scores_stats(
    per_image_scores: Tensor,
    images_classes: Tensor | None = None,
    only_class: int | None = None,
    outliers_policy: str | None = StatsOutliersPolicy.NONE,
    repeated_policy: str | None = StatsRepeatedPolicy.AVOID,
    repeated_replacement_atol: float = 1e-2,
) -> list[dict[str, str | int | float]]:
    """Torch-oriented interface for `per_image_scores_stats`. See its dscription for more details (below).

    Numpy version docstring
    =======================

    {docstring}
    """
    _validate.is_tensor(per_image_scores, "per_image_scores")
    per_image_scores_array = per_image_scores.detach().cpu().numpy()

    if images_classes is not None:
        _validate.is_tensor(images_classes, "images_classes")
        images_classes_array = images_classes.detach().cpu().numpy()

    else:
        images_classes_array = None

    # other validations happen inside `utils_numpy.per_image_scores_stats`

    return utils_numpy.per_image_scores_stats(
        per_image_scores_array,
        images_classes_array,
        only_class=only_class,
        outliers_policy=outliers_policy,
        repeated_policy=repeated_policy,
        repeated_replacement_atol=repeated_replacement_atol,
    )


per_image_scores_stats.__doc__ = per_image_scores_stats.__doc__.format(  # type: ignore[union-attr]
    docstring=utils_numpy.per_image_scores_stats.__doc__,
)
