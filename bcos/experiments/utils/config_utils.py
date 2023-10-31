"""
Utilities for working with configs.
"""
import argparse
import copy
import difflib
import warnings
from importlib import import_module
from typing import Any

try:
    from rich import print as pprint
except ImportError:
    from pprint import pprint

from .structure_constants import (
    ROOT,
    BASE_EXPERIMENTS_DIRECTORY,
    MODEL_FACTORY_MODULE,
    CONFIGS_VAR_NAME,
    CONFIGS_MODULE,
    MODEL_FACTORY_VAR_NAME,
)


__all__ = [
    "update_config",
    "configs_cli",
    "get_configs_and_model_factory",
    # specific config creators
    "create_configs_with_different_seeds",
]


def update_config(old_config, new_config):
    """Creates a new updated config.
    The old config is copied and updated with the newer one.
    This is done recursively.
    """
    result = copy.deepcopy(old_config)
    for k, v in new_config.items():
        # if subconfig update that recursively
        if k in result and isinstance(result[k], dict):
            assert isinstance(
                v, dict
            ), "Trying to overwrite a dict with something in a config!"
            result[k] = update_config(old_config=result[k], new_config=v)
        else:
            result[k] = v

    return result


def configs_cli(configs, *argv):
    parser = argparse.ArgumentParser(
        "Print config information. By default prints number of configs."
    )
    parser.add_argument(
        "-f",
        "--find",
        type=str,
        default=None,
        help="Check if given config is present and print it.",
    )
    parser.add_argument(
        "-a",
        "--print-all",
        action="store_true",
        default=False,
        help="Print all the names of the configs present.",
    )
    argv = None if len(argv) == 0 else argv
    args = parser.parse_args(argv)

    if len(configs) == 0:
        warnings.warn("No configs found. It's empty!")

    if args.find is not None:
        if args.find in configs:
            print(f"Found '{args.find}'")
            pprint(configs[args.find])
        else:
            print(f"No config named '{args.find}'!")
            maybe_alternative = difflib.get_close_matches(
                args.find, configs.keys(), n=1
            )
            if maybe_alternative:
                print(f"Did you mean '{maybe_alternative[0]}'?")
    elif args.print_all:
        for name in configs.keys():
            print(name)
    else:
        print(f"There are a total of {len(configs)} configs.")


def get_configs_and_model_factory(dataset, base_network):
    """
    Gets all the configs and the model factory function for given
    base network.

    Parameters
    ----------
    dataset : str
        The dataset for the model.
    base_network : str
        The base network.

    Returns
    -------
    A tuple of (configs, model_factory).
    """
    base_module_path = ".".join(
        [ROOT, BASE_EXPERIMENTS_DIRECTORY, dataset, base_network]
    )
    model_path = ".".join([base_module_path, MODEL_FACTORY_MODULE])
    configs_path = ".".join([base_module_path, CONFIGS_MODULE])

    try:
        model_module = import_module(model_path)
    except ModuleNotFoundError:
        print(f"Unable to import '{model_path}'")
        raise
    try:
        configs_module = import_module(configs_path)
    except ModuleNotFoundError:
        print(f"Unable to import '{configs_path}'")
        raise

    return getattr(configs_module, CONFIGS_VAR_NAME), getattr(
        model_module, MODEL_FACTORY_VAR_NAME
    )


# --------------------------------------------------------------------------------
# config creators
# --------------------------------------------------------------------------------
def create_configs_with_different_seeds(
    configs: "dict[str, dict[str, Any]]", seeds: "list[int] | int"
):
    """
    For each experiment config in given configs, changes the seed to given seed(s).

    Parameters
    ----------
    configs : dict[str, dict[str, Any]]
        The name to config dict.
    seeds: list[int] | int
        Alternatively provide multiple seeds.

    Returns
    -------
    dict[str, dict[str, Any]]
        Changed configs.
    """
    if isinstance(seeds, int):
        seeds = [seeds]

    result = dict()
    for seed in seeds:
        new_configs = copy.deepcopy(configs)
        new_configs = {f"{key}-{seed=}": value for key, value in new_configs.items()}
        for name, config in new_configs.items():
            config["seed"] = seed
        result.update(new_configs)

    return result
