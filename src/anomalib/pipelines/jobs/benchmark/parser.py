"""Config parser for benchmarking."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from argparse import SUPPRESS
from collections.abc import Iterable, Iterator, ValuesView
from itertools import product
from typing import Any

from jsonargparse import ArgumentParser, Namespace
from jsonargparse._optionals import get_doc_short_description
from jsonargparse._typehints import ActionTypeHint

from anomalib.data import AnomalibDataModule
from anomalib.models import AnomalyModule
from anomalib.pipelines.jobs.base import ConfigParser


class _GridSearchAction(ActionTypeHint):
    def __call__(self, *args, **kwargs) -> "_GridSearchAction | None":
        """Parse arguments for grid search."""
        if len(args) == 0:
            kwargs["_typehint"] = self._typehint
            kwargs["_enable_path"] = self._enable_path
            return _GridSearchAction(**kwargs)
        return None

    @staticmethod
    def _crawl(value: Any, parents: list[str], parent: str = "") -> None:  # noqa: ANN401
        """Crawl through the dictionary and store path to parents."""
        if isinstance(value, dict):
            for key, val in value.items():
                if key == "grid":
                    parents.append(parent)
                elif isinstance(val, dict):
                    parent_key = f"{parent}.{key}" if parent else key
                    _GridSearchAction._crawl(val, parents, parent_key)

    @staticmethod
    def pop_nested_key(container: dict, key: str) -> None:
        keys = key.split(".")
        if len(keys) > 1:
            _GridSearchAction.pop_nested_key(container[keys[0]], ".".join(keys[1:]))
        else:
            container.pop(keys[0])

    @staticmethod
    def sanitize_value(value: dict | Namespace) -> dict:
        """Returns a new value with all grid search keys removed."""
        _value = _GridSearchAction.recursive_dict(value) if isinstance(value, Namespace) else copy.deepcopy(value)
        keys: list[str] = []
        _GridSearchAction._crawl(_value, keys)
        for key in keys:
            _GridSearchAction.pop_nested_key(_value, key)
        return _value

    @staticmethod
    def recursive_namespace(container: dict) -> Namespace:
        """Convert dictionary to Namespace recursively."""
        output = Namespace()
        for k, v in container.items():
            if isinstance(v, dict):
                setattr(output, k, _GridSearchAction.recursive_namespace(v))
            else:
                setattr(output, k, v)
        return output

    @staticmethod
    def recursive_dict(container: Namespace) -> dict:
        """Convert Namespace to dictionary recursively."""
        output = {}
        for k, v in container.__dict__.items():
            if isinstance(v, Namespace):
                output[k] = _GridSearchAction.recursive_dict(v)
            else:
                output[k] = v
        return output

    def _check_type(self, value: dict | Namespace, append: bool = False, cfg: Namespace | None = None) -> Any:  # noqa: ANN401
        """Ignore all grid search keys.

        This allows the args to follow the same format as ``add_subclass_arguments``
        nested_key:
            class_path: str
            init_args: [arg1, arg2]
        at the same time allows grid search key
        nested_key:
            class_path:
                grid: [val1, val2]
            init_args:
                - arg1
                    grid: [val1, val2]
                - arg2
                    ...
        """
        _value = _GridSearchAction.sanitize_value(value)
        # check only the keys for which grid is not assigned.
        # this ensures that single keys are checked against the class
        _value = _GridSearchAction.recursive_namespace(_value)
        super()._check_type(_value, append, cfg)
        # convert original value to Namespace recursively
        value = _GridSearchAction.recursive_namespace(value)
        _GridSearchAction.discard_init_args_on_class_path_change(self, value, _value)
        _value.update(value)
        _GridSearchAction.apply_appends(self, _value)
        return _value


class _Parser(ConfigParser):
    """Parser for benchmarking job."""

    def __init__(self) -> None:
        super().__init__(name="Benchmarking parser")

    def _add_subclass_arguments(self, parser: ArgumentParser, baseclass: type, nested_key: str) -> None:
        """Adds the subclass of the provided class to the parser under nested_key."""
        doc_group = get_doc_short_description(baseclass, logger=parser.logger)
        group = parser._create_group_if_requested(  # noqa: SLF001
            baseclass,
            nested_key=nested_key,
            as_group=True,
            doc_group=doc_group,
            config_load=False,
            instantiate=False,
        )

        with _GridSearchAction.allow_default_instance_context():
            action = group.add_argument(
                f"--{nested_key}",
                metavar="CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE",
                help=(
                    'One or more arguments specifying "class_path"'
                    f' and "init_args" for any subclass of {baseclass.__name__}.'
                ),
                default=SUPPRESS,
                action=_GridSearchAction(typehint=baseclass, enable_path=True, logger=parser.logger),
            )
        action.sub_add_kwargs = {"fail_untyped": True, "sub_configs": True, "instantiate": True}

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add arguments to the parser."""
        parser.add_argument("--seed", type=int | dict[str, list[int]], default=42, help="Seed for reproducibility.")
        parser.add_argument(
            "--hardware",
            type=str,
            default="cuda",
            choices=["cpu", "cuda"],
            help="Hardware to run the benchmark on.",
        )
        self._add_subclass_arguments(parser, AnomalyModule, "model")
        self._add_subclass_arguments(parser, AnomalibDataModule, "data")

    @staticmethod
    def flatten_dict(config: dict, prefix: str = "") -> dict:
        """Flatten the dictionary."""
        out = {}
        for key, value in config.items():
            if isinstance(value, dict):
                out.update(_Parser.flatten_dict(value, f"{prefix}{key}."))
            else:
                out[f"{prefix}{key}"] = value
        return out

    @staticmethod
    def to_nested_dict(config: dict) -> dict:
        """Convert the flattened dictionary to nested dictionary."""
        out: dict[str, Any] = {}
        for key, value in config.items():
            keys = key.split(".")
            _dict = out
            for k in keys[:-1]:
                _dict = _dict.setdefault(k, {})
            _dict[keys[-1]] = value
        return out

    @staticmethod
    def convert_to_tuple(values: ValuesView) -> list[tuple]:
        """Convert a ValuesView object to a list of tuples.

        This is useful to get list of possible values for each parameter in the config and a tuple for values that are
        are to be patched. Ideally this is useful when used with product.

        Example:
            >>> params = DictConfig({
                    "dataset.category": [
                        "bottle",
                        "cable",
                    ],
                    "dataset.image_size": 224,
                    "model_name": ["padim"],
                })
            >>> convert_to_tuple(params.values())
            [('bottle', 'cable'), (224,), ('padim',)]
            >>> list(itertools.product(*convert_to_tuple(params.values())))
            [('bottle', 224, 'padim'), ('cable', 224, 'padim')]

        Args:
            values: ValuesView: ValuesView object to be converted to a list of tuples.

        Returns:
            list[Tuple]: List of tuples.
        """
        return_list = []
        for value in values:
            if isinstance(value, Iterable) and not isinstance(value, str):
                return_list.append(tuple(value))
            else:
                return_list.append((value,))
        return return_list

    def config_iterator(self, args: Namespace) -> Iterator:
        """Return iterator based on the arguments."""
        container = {
            "seed": args.seed,
            "hardware": args.hardware,
            "data": _GridSearchAction.recursive_dict(args.data),
            "model": _GridSearchAction.recursive_dict(args.model),
        }
        # extract all grid keys and return cross product of all grid values
        container = self.flatten_dict(container)
        grid_dict = {key: value for key, value in container.items() if "grid" in key}
        container = {key: value for key, value in container.items() if key not in grid_dict}
        combinations = list(product(*_Parser.convert_to_tuple(grid_dict.values())))
        for combination in combinations:
            _container = container.copy()
            for key, value in zip(grid_dict.keys(), combination, strict=True):
                _container[key.removesuffix(".grid")] = value
            yield _Parser.to_nested_dict(_container)
