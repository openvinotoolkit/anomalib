"""sigopt Logger
"""
from argparse import Namespace
from typing import Dict, Optional, Any, Union

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only
from sigopt.runs import RunFactoryProxyMethod

try:
    import sigopt
    import numpy as np
except ImportError:
    sigopt = None
    np = None


class SigoptLogger(LightningLoggerBase):
    def __init__(self, name: str, project: str, max_epochs: Optional[int] = 200, experiment=None):
        """Logger for sigopt

        :param name: Name of your run
        :param project: Name of your project
        :param max_epochs: The maximum number of epochs. Leave empty only if you are sure
            that your epochs won't go above 200 or your don't plan to use sigopt checkpoints, defaults to 200
        :param experiment: sigopt experiment, defaults to None
        """

        if sigopt is None:
            raise ImportError(
                "You want to use `sigopt` logger which is not installed yet,"  # pragma: no-cover
                " install it with `pip install sigopt`."
            )
        if np is None:
            raise ImportError("`numpy` dependency not met." " install it with `pip install numpy`.")  # pragma: no-cover

        super().__init__()

        self._name = name
        self._project = project

        # since only 200 checkpoints can be saved.
        # defaults to 1 if max_epochs > 200
        self._update_freq = max(1, np.ceil(max_epochs / 200))

        self._experiment = experiment

    @property
    @rank_zero_experiment
    def experiment(self) -> RunFactoryProxyMethod:
        if self._experiment is None:
            # even though this is called experiment this is sigopt run.
            # sigopt run is different from sigopt experiment
            self._experiment = sigopt.create_run(name=self._name, project=self._project)

        return self._experiment

    @rank_zero_only
    def log_image(self, image: np.ndarray, name: Optional[str] = None):
        """[summary]

        :param image: image in h w c (rgb/rgba) format. Supports multiple formats. See https://app.sigopt.com/docs/runs/reference#log_image
        :param name: Name of the image, defaults to None
        """
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        self.experiment.log_image(image, name)

    @rank_zero_only
    def log_checkpoint(self, metrics: Dict[str, float], epoch: int):
        """Logs the passed dict as a checkpoint. Has a limit of 200.
        Make sure that this is included in on_train_epoch_end

        param metrics: dictionary of metrics. Can only support maximum 4 metrics. This is a current limitation of sigopt
        :param epoch: current epoch
        :raises Exception: Raises exception if checkpoint number is greater than 200

        :Example:

        >>> def training_epoch_end(self, outputs: List[Any]) -> None:
        >>>     loss = 0
        >>>     for entry in outputs:
        >>>         loss += entry['loss']
        >>>         self.logger.log_checkpoint(metrics={"loss": loss/len(outputs)}, epoch=self.trainer.current_epoch)
        """

        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        # Use the epoch information to ensure that maximum of 200 checkpoints are saved
        if (epoch + 1) % self._update_freq != 0:
            return
        # uses checkpoint to save metrics. This way you will get the graph
        self.experiment.log_checkpoint(metrics)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int]):
        """Uses sigopt checkpoint to save the metrics. . This way you will get the graph.
            However it is unsafe as it does not check if number of checkpoints have crossed 200.

        :param metrics: Dictionary containing metrics. Current limitation is maximum of 4 metrics.
        :param step: trainer step
        """
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        try:
            self.experiment.log_checkpoint(metrics)
        except Exception as e:
            raise Exception(
                "Exception occurred. It is possible that you are trying to write more that 200 checkpoints. use `self.logger.log_checkpoint` for safer implementation"
            ) from e

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_callable_params(params)
        params = self._sanitize_other_params(params)
        self.experiment.set_parameters(params)

    @property
    def name(self) -> str:
        self._name

    @rank_zero_only
    def finalize(self, status):
        # end run
        self._experiment.end()

    @property
    def version(self) -> Optional[str]:
        return 1

    def _sanitize_other_params(self, params: Dict) -> Dict:
        """convert all params that are not string or numbers to string

        :param params: Flattened dictionary
        :return: Dict containing sanitized params
        """
        ret = {}
        for key, val in params.items():
            if not type(val) == int and not type(val) == float and not type(val) == str:
                val = str(val)
            ret[key] = val

        return ret
