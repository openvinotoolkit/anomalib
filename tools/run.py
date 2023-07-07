from argparse import ArgumentParser

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.trainer import AnomalibTrainer
from anomalib.utils.callbacks import get_callbacks


def train(name: str):
    config = get_configurable_parameters(model_name=name)
    datamodule = get_datamodule(config)
    model = get_model(config)
    callbacks = get_callbacks(config)
    print(callbacks)
    trainer = AnomalibTrainer(max_epochs=1, callbacks=callbacks)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    name = parser.parse_args().model
    train(name)
