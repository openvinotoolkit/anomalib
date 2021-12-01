"""AnomalibCLI."""
from typing import Optional

from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning.utilities.cli import OPTIMIZER_REGISTRY, LR_SCHEDULER_REGISTRY


class AnomalibArgumentParser(LightningArgumentParser):
    pass


class AnomalibCLI(LightningCLI):
    """AnomalibCLI [summary].

    Args:
        LightningCLI ([type]): [description]
    """

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds default arguments to the parser."""
        parser.add_argument(
            "--seed_everything",
            type=Optional[int],
            default=self.seed_everything_default,
            help="Set to an int to run seed_everything with this value before classes instantiation",
        )
        parser.add_argument("--algorithm", type=Optional[str], default="padim", help="Default anomaly algorithm.")
        parser.add_argument("--dataset", type=Optional[str], default="mvtec", help="Default dataset.")

    def _add_arguments(self, parser: LightningArgumentParser) -> None:
        # default + core + custom arguments
        self.add_default_arguments_to_parser(parser)
        self.add_core_arguments_to_parser(parser)
        self.add_arguments_to_parser(parser)
        # add default optimizer args if necessary
        if not parser._optimizers:  # already added by the user in `add_arguments_to_parser`
            parser.add_optimizer_args(OPTIMIZER_REGISTRY.classes)
        if not parser._lr_schedulers:  # already added by the user in `add_arguments_to_parser`
            parser.add_lr_scheduler_args(LR_SCHEDULER_REGISTRY.classes)
        self.link_optimizers_and_lr_schedulers(parser)

    def before_instantiate_classes(self) -> None:
        """Before Instantiate Classes."""
        print("done...")
