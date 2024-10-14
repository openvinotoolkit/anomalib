from lightning.pytorch.loggers import CSVLogger


class AnomalibCSVLogger(CSVLogger):
    def __init__(
            self,
            save_dir: str,
            name: str | None = "default",
            version: int | str | None = None,
            prefix: str = "",
            flush_logs_every_n_step: int | None = 100,
    ) -> None:
        super().__init__(
            save_dir=save_dir,
            name=name,
            version=version,
            prefix=prefix,
            flush_logs_every_n_steps=flush_logs_every_n_step,
        )
