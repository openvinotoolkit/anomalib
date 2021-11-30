"""AnomalibCLI."""
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI


class AnomalibCLI(LightningCLI):
    """AnomalibCLI [summary].

    Args:
        LightningCLI ([type]): [description]
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add Arguments to Parser."""
        parser.add_argument("--algorithm", default="padim")

    def before_instantiate_classes(self) -> None:
        """Before Instantiate Classes."""
        print("done...")
