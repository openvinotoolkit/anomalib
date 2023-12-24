"""Per-Image Overlap curve (PIMO, pronounced pee-mo) and its area under the curve (AUPIMO).

This module implements torch interfaces to access the numpy code in `pimo_numpy.py`.
Check its docstring for more details.

Tensors are build with `torch.from_numpy` and so the returned tensors will share the same memory as the numpy arrays.

Validations will preferably happen in ndarray so the numpy code can be reused without torch,
so often times the Tensor arguments will be converted to ndarray and then validated.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from torchmetrics import Metric

from anomalib.data.utils.image import duplicate_filename

from . import _validate, pimo_numpy
from .binclf_curve_numpy import Algorithm as BinclfAlgorithm
from .pimo_numpy import SharedFPRMetric

# =========================================== ARGS VALIDATION ===========================================


def _images_classes_from_masks(masks: Tensor) -> Tensor:
    masks = torch.concat(masks, dim=0)
    device = masks.device
    image_classes = pimo_numpy._images_classes_from_masks(masks.numpy())  # noqa: SLF001
    return torch.from_numpy(image_classes, device=device)


# =========================================== ARGS VALIDATION ===========================================


def _validate_anomaly_maps(anomaly_maps: Tensor) -> None:
    _validate.is_tensor(anomaly_maps, argname="anomaly_maps")
    _validate.anomaly_maps(anomaly_maps.numpy())


def _validate_masks(masks: Tensor) -> None:
    _validate.is_tensor(masks, argname="masks")
    _validate.masks(masks.numpy())


def _validate_threshs(threshs: Tensor) -> None:
    _validate.is_tensor(threshs, argname="threshs")
    _validate.threshs(threshs.numpy())


def _validate_shared_fpr(shared_fpr: Tensor, nan_allowed: bool = False, decreasing: bool = True) -> None:
    _validate.is_tensor(shared_fpr, argname="shared_fpr")
    _validate.rate_curve(shared_fpr.numpy(), nan_allowed=nan_allowed, decreasing=decreasing)


def _validate_image_classes(image_classes: Tensor) -> None:
    _validate.is_tensor(image_classes, argname="image_classes")
    _validate.image_classes(image_classes.numpy())


def _validate_per_image_tprs(per_image_tprs: Tensor, image_classes: Tensor) -> None:
    _validate.is_tensor(per_image_tprs, argname="per_image_tprs")
    _validate_image_classes(image_classes)

    _validate.per_image_rate_curves(
        per_image_tprs[image_classes == 1].numpy(),
        nan_allowed=False,
        decreasing=True,
    )

    normal_images_tprs = per_image_tprs[image_classes == 0]
    if not normal_images_tprs.isnan().all():
        msg = "Expected all normal images to have NaN TPRs, but some have non-NaN values."
        raise ValueError(msg)


def _validate_aupimos(aupimos: Tensor) -> None:
    _validate.is_tensor(aupimos, argname="aupimos")
    _validate.rates(aupimos.numpy(), nan_allowed=True)


# =========================================== RESULT OBJECT ===========================================


# TODO(jpcbertoldo): add image file path to `PIMOResult`  # noqa: TD003
# TODO(jpcbertoldo): missing docstring for `PIMOResult`  # noqa: TD003
@dataclass
class PIMOResult:  # noqa: D101
    # metadata
    shared_fpr_metric: str

    # data
    threshs: Tensor = field(repr=False)  # shape => (K,)
    shared_fpr: Tensor = field(repr=False)  # shape => (K,)
    per_image_tprs: Tensor = field(repr=False)  # shape => (N, K)

    @property
    def num_threshs(self) -> int:
        """Number of thresholds."""
        return self.threshs.shape[0]

    @property
    def num_images(self) -> int:
        """Number of images."""
        return self.per_image_tprs.shape[0]

    @property
    def image_classes(self) -> Tensor:
        """Image classes (0: normal, 1: anomalous)."""
        return (self.per_image_tprs.flatten(1) == 1).any(dim=1).to(torch.int32)

    def __post_init__(self) -> None:
        """Validate the inputs for the result object are consistent."""
        try:
            SharedFPRMetric.validate(self.shared_fpr_metric)
            _validate_threshs(self.threshs)
            _validate_shared_fpr(self.shared_fpr, nan_allowed=False)
            _validate_per_image_tprs(self.per_image_tprs, self.image_classes)

        except (TypeError, ValueError) as ex:
            msg = f"Invalid inputs for {self.__class__.__name__} object. Cause: {ex}."
            raise ValueError(msg) from ex

        if self.threshs.shape != self.shared_fpr.shape:
            msg = (
                f"Invalid {self.__class__.__name__} object. Attributes have inconsistent shapes: "
                f"threshs.shape={self.threshs.shape} != shared_fpr.shape={self.shared_fpr.shape}."
            )
            raise ValueError(msg)

        if self.threshs.shape[0] != self.per_image_tprs.shape[1]:
            msg = (
                f"Invalid {self.__class__.__name__} object. Attributes have inconsistent shapes: "
                f"threshs.shape[0]={self.threshs.shape[0]} != per_image_tprs.shape[1]={self.per_image_tprs.shape[1]}."
            )
            raise ValueError(msg)

    def thresh_at(self, fpr_level: float) -> tuple[int, float, float]:
        """Return the threshold at the given shared FPR.

        See `anomalib.utils.metrics.per_image.pimo_numpy.thresh_at_shared_fpr_level` for details.

        Args:
            fpr_level (float): shared FPR level

        Returns:
            tuple[int, float, float]:
                [0] index of the threshold
                [1] threshold
                [2] the actual shared FPR value at the returned threshold
        """
        return pimo_numpy.thresh_at_shared_fpr_level(
            self.threshs.numpy(),
            self.shared_fpr.numpy(),
            fpr_level,
        )

    def to_dict(self) -> dict[str, Tensor | str]:
        """Return a dictionary with the result object's attributes."""
        return {
            "shared_fpr_metric": self.shared_fpr_metric,
            "threshs": self.threshs,
            "shared_fpr": self.shared_fpr,
            "per_image_tprs": self.per_image_tprs,
        }

    @classmethod
    def from_dict(cls: type[PIMOResult], dic: dict[str, Tensor | str]) -> PIMOResult:
        """Return a result object from a dictionary."""
        keys = ["shared_fpr_metric", "threshs", "shared_fpr", "per_image_tprs"]
        for key in keys:
            if key not in dic:
                msg = f"Invalid input dictionary for {cls.__name__} object, missing key: {key}. Must contain: {keys}."
                raise ValueError(msg)

        return cls(**dic)

    def save(self, file_path: str | Path) -> None:
        """Save to a `.pt` file.

        Args:
            file_path: path to the `.pt` file where to save the PIMO result.
                - must have a `.pt` extension
                - if the file already exists, a numerical suffix is added to the filename
        """
        _validate.file_path(file_path, must_exist=False, extension=".pt")
        file_path = duplicate_filename(file_path)
        payload = self.to_dict()
        torch.save(payload, file_path)

    @classmethod
    def load(cls: type[PIMOResult], file_path: str | Path) -> PIMOResult:
        """Load from a `.pt` file.

        Args:
            file_path: path to the `.pt` file where to load the PIMO result.
                - must have a `.pt` extension
        """
        _validate.file_path(file_path, must_exist=True, extension=".pt")
        payload = torch.load(file_path)
        if not isinstance(payload, dict):
            msg = f"Invalid payload in file {file_path}. Must be a dictionary."
            raise TypeError(msg)
        try:
            return cls.from_dict(payload)
        except (TypeError, ValueError) as ex:
            msg = f"Invalid payload in file {file_path}. Cause: {ex}."
            raise ValueError(msg) from ex


# TODO(jpcbertoldo): add image file path to `AUPIMOResult`  # noqa: TD003
# TODO(jpcbertoldo): missing docstring for `AUPIMOResult`  # noqa: TD003
# TODO(jpcbertoldo): change `aucs` in the paper supp mat to `aupimos`  # noqa: TD003
@dataclass
class AUPIMOResult:  # noqa: D101
    # metadata
    shared_fpr_metric: str
    fpr_lower_bound: float
    fpr_upper_bound: float
    num_threshs: int

    # data
    thresh_lower_bound: float = field(repr=False)
    thresh_upper_bound: float = field(repr=False)
    aupimos: Tensor = field(repr=False)  # shape => (N,)

    @property
    def num_images(self) -> int:
        """Number of images."""
        return self.aupimos.shape[0]

    @property
    def image_classes(self) -> Tensor:
        """Image classes (0: normal, 1: anomalous)."""
        # if an instance has `nan` aupimo it's because it's a normal image
        return self.aupimos.isnan().to(torch.int32)

    @property
    def fpr_bounds(self) -> tuple[float, float]:
        """Lower and upper bounds of the FPR integration range."""
        return self.fpr_lower_bound, self.fpr_upper_bound

    @property
    def thresh_bounds(self) -> tuple[float, float]:
        """Lower and upper bounds of the threshold integration range.

        Recall: they correspond to the FPR bounds in reverse order.
        I.e.:
            fpr_lower_bound --> thresh_upper_bound
            fpr_upper_bound --> thresh_lower_bound
        """
        return self.thresh_lower_bound, self.thresh_upper_bound

    def __post_init__(self) -> None:
        """Validate the inputs for the result object are consistent."""
        try:
            SharedFPRMetric.validate(self.shared_fpr_metric)
            _validate.rate_range((self.fpr_lower_bound, self.fpr_upper_bound))
            _validate.num_threshs(self.num_threshs)
            _validate_aupimos(self.aupimos)
            _validate.thresh_bounds((self.thresh_lower_bound, self.thresh_upper_bound))

        except (TypeError, ValueError) as ex:
            msg = f"Invalid inputs for {self.__class__.__name__} object. Cause: {ex}."
            raise ValueError(msg) from ex

    @classmethod
    def from_pimoresult(
        cls: type[AUPIMOResult],
        pimoresult: PIMOResult,
        fpr_bounds: tuple[float, float],
        aupimos: Tensor,
    ) -> AUPIMOResult:
        """Return an AUPIMO result object from a PIMO result object.

        Args:
            pimoresult: PIMO result object
            fpr_bounds: lower and upper bounds of the FPR integration range
            aupimos: AUPIMO scores
        """
        if pimoresult.per_image_tprs.shape[0] != aupimos.shape[0]:
            msg = (
                f"Invalid {cls.__name__} object. Attributes have inconsistent shapes: "
                f"there are {pimoresult.per_image_tprs.shape[0]} PIMO curves but {aupimos.shape[0]} AUPIMO scores."
            )
            raise ValueError(msg)

        if not torch.isnan(aupimos[pimoresult.image_classes == 0]).all():
            msg = "Expected all normal images to have NaN AUPIMOs, but some have non-NaN values."
            raise ValueError(msg)

        if torch.isnan(aupimos[pimoresult.image_classes == 1]).any():
            msg = "Expected all anomalous images to have valid AUPIMOs (not nan), but some have NaN values."
            raise ValueError(msg)

        fpr_lower_bound, fpr_upper_bound = fpr_bounds
        # recall: fpr upper/lower bounds are the same as the thresh lower/upper bounds
        _, thresh_lower_bound, __ = pimoresult.thresh_at(fpr_upper_bound)
        _, thresh_upper_bound, __ = pimoresult.thresh_at(fpr_lower_bound)
        # `_` is the threshold's index, `__` is the actual fpr value
        return cls(
            shared_fpr_metric=pimoresult.shared_fpr_metric,
            fpr_lower_bound=fpr_lower_bound,
            fpr_upper_bound=fpr_upper_bound,
            num_threshs=pimoresult.num_threshs,
            thresh_lower_bound=float(thresh_lower_bound),
            thresh_upper_bound=float(thresh_upper_bound),
            aupimos=aupimos,
        )

    def to_dict(self) -> dict[str, Tensor | str | float | int]:
        """Return a dictionary with the result object's attributes."""
        return {
            "shared_fpr_metric": self.shared_fpr_metric,
            "fpr_lower_bound": self.fpr_lower_bound,
            "fpr_upper_bound": self.fpr_upper_bound,
            "num_threshs": self.num_threshs,
            "thresh_lower_bound": self.thresh_lower_bound,
            "thresh_upper_bound": self.thresh_upper_bound,
            "aupimos": self.aupimos,
        }

    @classmethod
    def from_dict(cls: type[AUPIMOResult], dic: dict[str, Tensor | str | float | int]) -> AUPIMOResult:
        """Return a result object from a dictionary."""
        keys = [
            "shared_fpr_metric",
            "fpr_lower_bound",
            "fpr_upper_bound",
            "num_threshs",
            "thresh_lower_bound",
            "thresh_upper_bound",
            "aupimos",
        ]
        for key in keys:
            if key not in dic:
                msg = f"Invalid input dictionary for {cls.__name__} object, missing key: {key}. Must contain: {keys}."
                raise ValueError(msg)

        return cls(**dic)  # type: ignore[arg-type]

    def save(self, file_path: str | Path) -> None:
        """Save to a `.json` file.

        Args:
            file_path: path to the `.json` file where to save the AUPIMO result.
                - must have a `.json` extension
                - if the file already exists, a numerical suffix is added to the filename
        """
        _validate.file_path(file_path, must_exist=False, extension=".json")
        file_path = duplicate_filename(file_path)
        file_path = Path(file_path)
        payload = self.to_dict()
        aupimos: Tensor = payload["aupimos"]
        payload["aupimos"] = aupimos.numpy().tolist()
        with file_path.open("w") as f:
            json.dump(payload, f, indent=4)

    @classmethod
    def load(cls: type[AUPIMOResult], file_path: str | Path) -> AUPIMOResult:
        """Load from a `.json` file.

        Args:
            file_path: path to the `.json` file where to load the AUPIMO result.
                - must have a `.json` extension
        """
        _validate.file_path(file_path, must_exist=True, extension=".json")
        file_path = Path(file_path)
        with file_path.open("r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            file_path = str(file_path)
            msg = f"Invalid payload in file {file_path}. Must be a dictionary."
            raise TypeError(msg)
        payload["aupimos"] = torch.tensor(payload["aupimos"], dtype=torch.float64)
        try:
            return cls.from_dict(payload)
        except (TypeError, ValueError) as ex:
            msg = f"Invalid payload in file {file_path}. Cause: {ex}."
            raise ValueError(msg) from ex


# =========================================== FUNCTIONAL ===========================================


# TODO(jpcbertoldo): missing docstring for `pimo`  # noqa: TD003
def pimo_curves(  # noqa: D103
    anomaly_maps: Tensor,
    masks: Tensor,
    num_threshs: int,
    binclf_algorithm: str = BinclfAlgorithm.NUMBA,
    shared_fpr_metric: str = SharedFPRMetric.MEAN_PERIMAGE_FPR,
) -> PIMOResult:
    _validate.is_tensor(anomaly_maps, argname="anomaly_maps")
    anomaly_maps_array = anomaly_maps.detach().cpu().numpy()

    _validate.is_tensor(masks, argname="masks")
    masks_array = masks.detach().cpu().numpy()

    # other validations are done in the numpy code
    threshs_array, shared_fpr_array, per_image_tprs_array, _ = pimo_numpy.pimo_curves(
        anomaly_maps_array,
        masks_array,
        num_threshs,
        binclf_algorithm=binclf_algorithm,
        shared_fpr_metric=shared_fpr_metric,
    )
    # _ is `image_classes` -- not needed here because it's a property in the result object

    # tensors are build with `torch.from_numpy` and so the returned tensors
    # will share the same memory as the numpy arrays
    device = anomaly_maps.device
    # N: number of images, K: number of thresholds
    # shape => (K,)
    threshs = torch.from_numpy(threshs_array).to(device)
    # shape => (K,)
    shared_fpr = torch.from_numpy(shared_fpr_array).to(device)
    # shape => (N, K)
    per_image_tprs = torch.from_numpy(per_image_tprs_array).to(device)

    return PIMOResult(
        shared_fpr_metric=shared_fpr_metric,
        threshs=threshs,
        shared_fpr=shared_fpr,
        per_image_tprs=per_image_tprs,
    )


# TODO(jpcbertoldo): missing docstring for `aupimo`  # noqa: TD003
def aupimo_scores(  # noqa: D103
    anomaly_maps: Tensor,
    masks: Tensor,
    num_threshs: int = 300_000,
    binclf_algorithm: str = BinclfAlgorithm.NUMBA,
    shared_fpr_metric: str = SharedFPRMetric.MEAN_PERIMAGE_FPR,
    fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
    force: bool = False,
) -> tuple[PIMOResult, AUPIMOResult]:
    _validate.is_tensor(anomaly_maps, argname="anomaly_maps")
    anomaly_maps_array = anomaly_maps.detach().cpu().numpy()

    _validate.is_tensor(masks, argname="masks")
    masks_array = masks.detach().cpu().numpy()

    # other validations are done in the numpy code

    threshs_array, shared_fpr_array, per_image_tprs_array, _, aupimos_array = pimo_numpy.aupimo_scores(
        anomaly_maps_array,
        masks_array,
        num_threshs,
        binclf_algorithm=binclf_algorithm,
        shared_fpr_metric=shared_fpr_metric,
        fpr_bounds=fpr_bounds,
        force=force,
    )

    # tensors are build with `torch.from_numpy` and so the returned tensors
    # will share the same memory as the numpy arrays
    device = anomaly_maps.device
    # N: number of images, K: number of thresholds
    # shape => (K,)
    threshs = torch.from_numpy(threshs_array).to(device)
    # shape => (K,)
    shared_fpr = torch.from_numpy(shared_fpr_array).to(device)
    # shape => (N, K)
    per_image_tprs = torch.from_numpy(per_image_tprs_array).to(device)
    # shape => (N,)
    aupimos = torch.from_numpy(aupimos_array).to(device)

    pimoresult = PIMOResult(
        shared_fpr_metric=shared_fpr_metric,
        threshs=threshs,
        shared_fpr=shared_fpr,
        per_image_tprs=per_image_tprs,
    )
    aupimoresult = AUPIMOResult.from_pimoresult(
        pimoresult,
        fpr_bounds=fpr_bounds,
        aupimos=aupimos,
    )
    return pimoresult, aupimoresult


# =========================================== TORCHMETRICS ===========================================


# TODO(jpcbertoldo): missing docstring for `PIMO`  # noqa: TD003
class PIMO(Metric):  # noqa: D101
    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    num_threshs: int
    binclf_algorithm: str
    shared_fpr_metric: str

    anomaly_maps: list[Tensor]
    masks: list[Tensor]

    @property
    def is_empty(self) -> bool:
        """Return True if the metric has not been updated yet."""
        return len(self.anomaly_maps) == 0

    @property
    def num_images(self) -> int:
        """Number of images."""
        return sum([am.shape[0] for am in self.anomaly_maps])

    @property
    def image_classes(self) -> Tensor:
        """Image classes (0: normal, 1: anomalous)."""
        return _images_classes_from_masks(self.masks)

    def __init__(
        self,
        num_threshs: int,
        binclf_algorithm: str = BinclfAlgorithm.NUMBA,
        shared_fpr_metric: str = SharedFPRMetric.MEAN_PERIMAGE_FPR,
    ) -> None:
        """Per-Image Overlap (PIMO) curve."""
        # TODO(jpcbertoldo): docstring of `PIMO.__init__()`  # noqa: TD003
        super().__init__()

        warnings.warn(
            f"Metric `{self.__class__.__name__}` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
            stacklevel=1,
        )

        # the options below are, redundantly, validated here to avoid reaching
        # an error later in the execution

        _validate.num_threshs(num_threshs)
        self.num_threshs = num_threshs

        # validate binclf_algorithm and shared_fpr_metric
        BinclfAlgorithm.validate(binclf_algorithm)
        self.binclf_algorithm = binclf_algorithm

        SharedFPRMetric.validate(shared_fpr_metric)
        self.shared_fpr_metric = SharedFPRMetric.MEAN_PERIMAGE_FPR

        self.add_state("anomaly_maps", default=[], dist_reduce_fx="cat")
        self.add_state("masks", default=[], dist_reduce_fx="cat")

    def update(self, anomaly_maps: Tensor, masks: Tensor) -> None:
        """Update list of anomaly maps and masks.

        Args:
            anomaly_maps (Tensor): predictions of the model (ndim == 2, float)
            masks (Tensor): ground truth masks (ndim == 2, binary)
        """
        _validate_anomaly_maps(anomaly_maps)
        _validate_masks(masks)
        _validate.same_shape(anomaly_maps, masks)
        self.anomaly_maps.append(anomaly_maps)
        self.masks.append(masks)

    # TODO(jpcbertoldo): missing docstring for `PIMO.compute`  # noqa: TD003
    def compute(self) -> PIMOResult:  # noqa: D102
        if self.is_empty:
            msg = "No anomaly maps and masks have been added yet. Please call `update()` first."
            raise RuntimeError(msg)
        anomaly_maps = torch.concat(self.anomaly_maps, dim=0)
        masks = torch.concat(self.masks, dim=0)
        return pimo_curves(
            anomaly_maps,
            masks,
            self.num_threshs,
            binclf_algorithm=self.binclf_algorithm,
            shared_fpr_metric=self.shared_fpr_metric,
        )


class AUPIMO(PIMO):
    """Area Under the Per-Image Overlap (PIMO) curve.

    TODO(jpcbertoldo): docstring of `AUPIMO`  # noqa: DAR101
    """

    fpr_bounds: tuple[float, float]
    force: bool

    @staticmethod
    def normalizing_factor(fpr_bounds: tuple[float, float]) -> float:
        """Constant that normalizes the AUPIMO integral to 0-1 range.

        It is the maximum possible value from the integral in AUPIMO's definition.
        It corresponds to assuming a constant function T_i: thresh --> 1.

        Args:
            fpr_bounds: lower and upper bounds of the FPR integration range.

        Returns:
            float: the normalization factor (>0).
        """
        return pimo_numpy.aupimo_normalizing_factor(fpr_bounds)

    @staticmethod
    def random_model_score(fpr_bounds: tuple[float, float]) -> float:
        """AUPIMO of a theoretical random model.

        "Random model" means that there is no discrimination between normal and anomalous pixels/patches/images.
        It corresponds to assuming the functions T = F.

        For the FPR bounds (1e-5, 1e-4), the random model AUPIMO is ~4e-5.

        Args:
            fpr_bounds: lower and upper bounds of the FPR integration range.

        Returns:
            float: the AUPIMO score.
        """
        return pimo_numpy.aupimo_random_model_score(fpr_bounds)

    def __repr__(self) -> str:
        """Show the metric name and its integration bounds."""
        metric = self.shared_fpr_metric
        lower, upper = self.fpr_bounds
        return f"{self.__class__.__name__}({metric} in [{lower:.2g}, {upper:.2g}])"

    def __init__(
        self,
        num_threshs: int = 300_000,
        binclf_algorithm: str = BinclfAlgorithm.NUMBA,
        shared_fpr_metric: str = SharedFPRMetric.MEAN_PERIMAGE_FPR,
        fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
        force: bool = False,
    ) -> None:
        """Area Under the Per-Image Overlap (PIMO) curve.

        TODO(jpcbertoldo): docstring of `AUPIMO.__init__()`  # noqa: DAR101
        """
        super().__init__(
            num_threshs=num_threshs,
            binclf_algorithm=binclf_algorithm,
            shared_fpr_metric=shared_fpr_metric,
        )

        # other validations are done in PIMO.__init__()

        _validate.rate_range(fpr_bounds)
        self.fpr_bounds = fpr_bounds

        self.force = force

    def compute(self, force: bool | None = None) -> tuple[PIMOResult, AUPIMOResult]:  # type: ignore[override]
        """TODO(jpcbertoldo): docstring of `AUPIMO.compute()`."""  # noqa: D402
        if self.is_empty:
            msg = "No anomaly maps and masks have been added yet. Please call `update()` first."
            raise RuntimeError(msg)
        anomaly_maps = torch.concat(self.anomaly_maps, dim=0)
        masks = torch.concat(self.masks, dim=0)
        force = force if force is not None else self.force
        return aupimo_scores(
            anomaly_maps,
            masks,
            self.num_threshs,
            binclf_algorithm=self.binclf_algorithm,
            shared_fpr_metric=self.shared_fpr_metric,
            fpr_bounds=self.fpr_bounds,
            force=force,
        )
