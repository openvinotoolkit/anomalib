from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from sigopt import Connection

from anomalib.config.config import get_configurable_parameters
from anomalib.config.sweep_config import get_experiment
from anomalib.datasets import get_datamodule
from anomalib.models import get_model


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--sweep_config_path", type=str, default="hparams_configs/stfpm_sweep.yaml", required=False, help="Path to hyperparameter sweep config file.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    config = get_configurable_parameters(model_name=args.model, model_config_path=args.model_config_path)
    
    connection = Connection()
    experiment = get_experiment(connection=connection, sweep_config_path=args.sweep_config_path)
    print("Created: https://app.sigopt.com/experiment/" + experiment.id)

    if config.project.seed != 0:
        seed_everything(config.project.seed)

    datamodule = get_datamodule(config)

    while experiment.progress.observation_count < experiment.observation_budget:
        suggestion = connection.experiments(experiment.id).suggestions().create()

        # get names of the suggested params
        params = suggestion.assignments.keys()
        # replace the params in config with suggested param values
        for param in params:
            config.model[param] = suggestion.assignments[param]

        model = get_model(config)

        trainer = Trainer(callbacks=model.callbacks, **config.trainer)
        trainer.fit(model=model, datamodule=datamodule)
        result = trainer.test(model=model, datamodule=datamodule)

        # get the value of the required metric from the results
        for metrics in result:
            if experiment.metric.name in metrics.keys():
                value = metrics[experiment.metric.name]

        if value is None:
            raise ValueError(f"Model does not return {experiment.metric.name} from test step")
        
        # upload experiment result
        connection.experiments(experiment.id).observations().create(
            suggestion = suggestion.id,
            value = value
        )
        experiment = connection.experiments(experiment.id).fetch()
