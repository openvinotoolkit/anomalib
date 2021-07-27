"""sigopt Logger
"""
from argparse import Namespace
from typing import Any, Dict, Optional, Union

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

# TODO refactor import check https://jira.devtools.intel.com/browse/IAAALD-24
try:
    import sigopt
    from sigopt.exception import ApiException
    from sigopt.runs import RunFactoryProxyMethod
except ImportError as e:
    raise ImportError(
        "You want to use `sigopt` logger which is not installed yet, install it with `pip install sigopt`."
    ) from e
try:
    import numpy as np
except ImportError as e:
    raise ImportError("`numpy` dependency not met. Install it with `pip install numpy`.") from e


class SigoptLogger(LightningLoggerBase):
    def __init__(self, name: str, project: str, max_epochs: Optional[int] = 200, experiment=None):
        """Logger for sigopt

        Args:
            name: Name of your run
            project: Name of your project
            max_epochs: The maximum number of epochs. Leave empty only if you are sure
            that your epochs won't go above 200 or your don't plan to use sigopt checkpoints, defaults to 200
            experiment: sigopt experiment, defaults to None
        """
        super().__init__()

        self._name = name
        self._project = project

        # since only 200 checkpoints can be saved.
        self._update_freq: int = max(1, np.ceil(max_epochs / 200))

        self._experiment = experiment

    @property
    @rank_zero_experiment
    def experiment(self) -> RunFactoryProxyMethod:
        """Create experiment object"""
        if self._experiment is None:
            # even though this is called experiment this is sigopt run.
            # sigopt run is different from sigopt experiment
            self._experiment = sigopt.create_run(name=self._name, project=self._project)

        return self._experiment

    @rank_zero_only
    def log_image(self, image: np.ndarray, name: Optional[str] = None):
        """Logs images

        Args:
          image: np.ndarray: image in h w c (rgb/rgba) format. Supports multiple formats.
            See https://app.sigopt.com/docs/runs/reference#log_image
          name: Optional[str]: Name of the image, defaults to None

        """
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        self.experiment.log_image(image, name)

    @rank_zero_only
    def log_checkpoint(self, metrics: Dict[str, float], epoch: int) -> None:
        """Logs the passed dict as a checkpoint. Has a limit of 200.
        Make sure that this is included in on_train_epoch_end

        Args:
          epoch: int: Current training epoch. This is needed to calculate logging frequency
          metrics: Dict[str, float]:  dictionary of metrics. Can only support maximum 4 metrics.
           This is a current limitation of sigopt.

        Raises:
          Exception: Raises exception if checkpoint number is greater than 200

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
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Uses sigopt checkpoint to save the metrics. This way you will get the graph.
            However it is unsafe as it does not check if number of checkpoints have crossed 200.

        Args:
          metrics: Dictionary containing metrics. Current limitation is maximum of 4 metrics.
            Also uses `log_metric` to log other metrics
          step: trainer step. Not used here
        """
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        try:
            self.experiment.log_checkpoint(metrics)
            for k, v in metrics.items():
                self.experiment.log_metric(name=k, value=v)
        except ApiException as e:
            raise ValueError(
                "Exception occurred."
                "It is possible that you are trying to write more that 200 checkpoints."
                "Use `self.logger.log_checkpoint` for safer implementation"
            ) from e

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        """

        Args:
          params: Union[Dict[str, Any] | Namespace]: Logs the model hyperparameters.
        """
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_callable_params(params)
        params = self._sanitize_other_params(params)

        # SigOpt allows only 100 keys at a time. This is a simple fix which splits the values into chunks of 100
        if len(params) > 100:
            entries = list(params.items())
            for i in range(0, len(params), 100):
                subset = dict(entries[i : i + 100])
                self.experiment.set_parameters(subset)
        else:
            self.experiment.set_parameters(params)

    @property
    def name(self) -> str:
        """returns name of the experiment"""
        return self._name

    @rank_zero_only
    def finalize(self, status) -> None:
        """Closes the experiment object

        Args:
          status: Not used

        """
        self._experiment.end()

    @property
    def version(self) -> int:
        """Added for PytorchLogger compatibility"""
        return 1

    @staticmethod
    def _sanitize_other_params(params: Dict) -> Dict:
        """convert all params that are not string or numbers to string

        Args:
          params: Dict: Flattened dictionary

        Returns:
          Dict containing sanitized params

        """
        ret = {}
        for key, val in params.items():
            # isinstance is not used for bool type as it returns true for int
            if type(val) != int and not isinstance(val, float) and not isinstance(val, str):
                val = str(val)
            ret[str(key)] = val  # sanitize keys as well

        return ret
