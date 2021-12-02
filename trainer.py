"""Trainer."""

from anomalib.cli import AnomalibCLI

cli = AnomalibCLI()


cli.trainer.fit(datamodule=cli.datamodule, model=cli.model)
cli.trainer.test(datamodule=cli.datamodule, model=cli.model)
