"""
Utilities for loading models, configs, etc.
"""
import difflib
import warnings
from pathlib import Path
from typing import Literal, Union, Optional, Any

import torch

from .config_utils import get_configs_and_model_factory, update_config
from .structure_constants import ROOT, BASE_EXPERIMENTS_DIRECTORY

__all__ = [
    "Experiment",  # main class
    "Metrics",
    # exceptions
    "EMAWeightsNotFound",
    "EmptyModelStateDictError",
]


class Metrics(dict):
    """
    A dictionary for storing configs with additional helper methods.
    """

    VALIDATION_KEY = "eval_acc1"
    EMA_VALIDATION_KEY = "eval_acc1_ema"

    def __init__(self, other: dict):
        import torch

        metrics = {name: torch.tensor(value) for name, value in other.items()}
        super().__init__(metrics)

    @property
    def best_epoch_and_accuracy(self) -> tuple[int, float]:
        """A tuple containing the best epoch and accuracy
        w.r.t. the top-1 validation accuracy."""

        return self.get_best_epoch_and_metric_value_for(self.VALIDATION_KEY)

    @property
    def best_epoch_and_accuracy_ema(self) -> tuple[int, float]:
        """A tuple containing the best epoch and accuracy
        w.r.t. the top-1 validation accuracy."""
        assert self.EMA_VALIDATION_KEY in self, "EMA metrics not found!"

        return self.get_best_epoch_and_metric_value_for(self.EMA_VALIDATION_KEY)

    def get_best_epoch_and_metric_value_for(
        self,
        metric_key: str,
        mode: Literal["min", "max"] = "max",
    ) -> tuple[int, float]:
        """
        Get the best epoch and metric value in metric collection
        for given metric key/name.

        Parameters
        ----------
        metric_key : str
            The metric key/name to search in.
        mode : Literal["min", "max"]
            Whether to min. or max. to find best. Default: max.

        Returns
        -------
        tuple[int, float]
            The epoch and best metric value, respectively.
        """

        metric_values = self[metric_key][:, 1]
        if mode == "max":
            best_epoch_idx = metric_values.argmax()
        elif mode == "min":
            best_epoch_idx = metric_values.argmin()
        else:
            raise ValueError(f"Unknown {mode=}")

        best_entry = self[metric_key][best_epoch_idx]
        best_epoch = int(best_entry[0])
        best_metric_value = float(best_entry[1])
        return best_epoch, best_metric_value


class Experiment:
    """
    Utility class for loading models and configs.
    """

    MODEL_STATE_DICT_PREFIX = "model."
    MODEL_STATE_DICT_EMA_PREFIX = "ema.module."

    def __init__(
        self,
        path_or_dataset: Union[str, Path],
        base_network: Optional[str] = None,
        experiment_name: Optional[str] = None,
        base_directory: str = BASE_EXPERIMENTS_DIRECTORY,
    ):
        """
        Initialize an Experiment object.

        Parameters
        ----------
        path_or_dataset : Union[str, Path]
            The path to a stored experiment OR the name of the dataset.
        base_network : Optional[str]
            The base network. Only provide this if `path_or_dataset` is a dataset.
        experiment_name : Optional[str]
            The experiment name. Only provide this if `path_or_dataset` is a dataset.
        base_directory : str
            The base directory. Default: "./experiments"
        """
        if "/" in path_or_dataset:  # will assume it's a path
            assert (
                base_network is None and experiment_name is None
            ), f"Path provided ('{path_or_dataset=}' b/c it contains '/')! Other arguments should be None!"
            path = Path(path_or_dataset)
            experiment_name = path.name
            base_network = path.parent.name
            dataset = path.parent.parent.name
            base_directory = path.parent.parent.parent
        else:  # assume it's a dataset
            dataset = path_or_dataset
            DATASETS = [
                p.name
                for p in Path(ROOT, BASE_EXPERIMENTS_DIRECTORY).iterdir()
                if p.is_dir()
            ]
            if dataset not in DATASETS:
                raise ValueError(
                    "Since no '/' found in first argument, assumed it to be a "
                    f"dataset. However, no such dataset ('{dataset}') exists!"
                )
            assert (
                base_network is not None and experiment_name is not None
            ), "Provide other arguments too!"

        configs, get_model = get_configs_and_model_factory(
            dataset, base_network)

        self._get_model = get_model

        if experiment_name not in configs:
            msg = f"Config for '{experiment_name=}' not found!"
            maybe = difflib.get_close_matches(experiment_name, configs.keys())
            if maybe:
                msg += f" Did you mean '{maybe[0]}'?"
            raise ValueError(msg)

        self.config = configs[experiment_name]
        self.base_directory = base_directory
        self.dataset = dataset
        self.base_network = base_network
        self.experiment_name = experiment_name
        self.save_dir = Path(base_directory, dataset,
                             base_network, experiment_name)

    @property
    def datamodule(self):
        """The PytorchLightning datamodule for the experiment.
        Need to set up before you can use. E.g.:

        Example
        -------
        >>> exp = Experiment(...)
        >>> datamodule = exp.datamodule
        >>> datamodule.setup("val")
        >>> val_dataloader = datamodule.val_dataloader()
        """
        from bcos.data.datamodules import CIFAR10DataModule, ImageNetDataModule

        if self.dataset == "CIFAR10":
            datamodule = CIFAR10DataModule(self.config["data"])
        elif self.dataset == "ImageNet":
            datamodule = ImageNetDataModule(self.config["data"])
        else:
            raise ValueError(f"Unknown dataset: '{self.dataset}'")

        return datamodule

    @property
    def last_checkpoint_path(self) -> Path:
        """
        Get the path of the last checkpoint for this experiment.
        """
        return self.save_dir / "last.ckpt"

    @property
    def last_checkpoint_state_dict(self) -> dict[str, Any]:
        """
        Try to load the last checkpoint state dict.
        """
        save_path = self.last_checkpoint_path
        return self._load_state_dict_from_path(save_path)

    def get_model(self, **kwargs: Any):
        """
        Create a new instance of the model for the experiment.
        """
        model_config = self.config["model"]
        model_config = update_config(model_config, kwargs)
        return self._get_model(model_config)

    def load_metrics(self) -> Metrics:
        """
        Returns
        -------
        Metrics
            Load a dictionary containing metrics from the last state dict.
            Tensor shape: [num_epochs (+1), 2] with columns (#epoch, metric_value)
            You can use the `.best_epoch_and_accuracy` attribute of the returned
            dict to get tuple containing the best epoch and accuracy. (See also
            `.get_best_epoch_and_metric_value_for(key)` method.)
        """
        state_dict = self.last_checkpoint_state_dict
        return self._load_metrics(state_dict)

    def load_pretrained_model(
        self,
        reload: str = "last",
        verbose: bool = False,
        ema: bool = False,
        return_pl_state_dict: bool = False,
    ) -> Union[torch.nn.Module, tuple[torch.nn.Module, dict[str, Any]]]:
        """
        Load a trained model.

        Parameters
        ----------
        reload : str
            What weights to reload. Either "last", "best" or "epoch_<N>".
            Default: "last"
        verbose : bool
            Print information about loaded epoch and top-1 accuracy.
            Default: False
        ema : bool
            Load the EMA weights.
        return_pl_state_dict : bool
            Whether to ALSO return the whole PL state dict instead.
            Default: False

        Returns
        -------
        Union[torch.nn.Module, tuple[torch.nn.Module, dict[str, Any]]]
            A trained model.
            Or a trained model along with its complete PL state dict.
        """
        model = self.get_model()
        pl_state_dict = self.last_checkpoint_state_dict

        # first depending on the reload get the specific pl state dict
        if reload == "last":
            pass
        elif reload == "best":
            metrics = self._load_metrics(pl_state_dict)
            best_epoch, _ = (
                metrics.best_epoch_and_accuracy_ema
                if ema
                else metrics.best_epoch_and_accuracy
            )

            try:
                best_save_path = next(self.save_dir.glob(
                    f"epoch={best_epoch}-*.ckpt"))
            except StopIteration:
                raise FileNotFoundError(
                    f"No ckpt found for {best_epoch=} in {self.save_dir}"
                )
            pl_state_dict = self._load_state_dict_from_path(best_save_path)
        elif reload.startswith("epoch_"):
            reload = reload.replace("_", "=")

            try:
                save_path = next(self.save_dir.glob(f"{reload}-*.ckpt"))
            except StopIteration:
                raise FileNotFoundError(
                    f"No ckpt found for {reload=} in {self.save_dir}"
                )

            pl_state_dict = self._load_state_dict_from_path(save_path)
        else:
            raise ValueError(f"Unknown reload type: '{reload}'")

        # now load the standard or ema weights from the state dict
        if not ema:
            self.model_load_state_dict(
                model, state_dict=pl_state_dict["state_dict"])
        else:
            prefix = self.MODEL_STATE_DICT_EMA_PREFIX
            try:
                self.model_load_state_dict(
                    model, state_dict=pl_state_dict["state_dict"], prefix=prefix
                )
            except EmptyModelStateDictError:
                raise EMAWeightsNotFound(
                    "Unable to find EMA weights in given PL state dict!"
                )

        # print extra information if need be
        if verbose:
            epoch = pl_state_dict["epoch"]
            ema_suffix = "(EMA)" if ema else ""
            print(f"Loaded epoch: {epoch} {ema_suffix}")
            metrics = self._load_metrics(pl_state_dict)
            val_acc = None
            for epoch_i, val_acc_i in metrics["eval_acc1"]:
                if epoch_i == epoch:
                    val_acc = val_acc_i
                    break
            assert val_acc is not None, "Unable to find val acc!"
            print(f"With validation accuracy: {val_acc:.2%}")

        if return_pl_state_dict:
            return model, pl_state_dict
        else:
            return model

    @property
    def available_checkpoints(self) -> list[Path]:
        """Returns a list to paths to available checkpoint files if any."""
        if not self.save_dir.exists():
            warnings.warn(
                f"Empty directory, no checkpoints found! {self.save_dir}")
            return []
        return list(p for p in self.save_dir.iterdir() if p.suffix == ".ckpt")

    @staticmethod
    def model_load_state_dict(
        model, state_dict, prefix=MODEL_STATE_DICT_PREFIX
    ) -> None:
        """
        Load given state dict into model. This supports removing a given prefix
        from the keys (which also acts as the filter).

        Parameters
        ----------
        model : torch.nn.Module
            The model to put the state into.
        state_dict : dict[str, Any]
            The dict containing the state.
        prefix: str
            The prefix to remove from the state dict's keys.
            This acts as a filter. (useful if ema etc.)
            Default: "model."
        """
        N = len(prefix) if prefix else 0
        model_state_dict = {
            key[N:]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if len(model_state_dict) == 0 and prefix != "":
            raise EmptyModelStateDictError(
                f"Filter '{prefix}' resulted in an empty model state dict!"
            )
        model.load_state_dict(model_state_dict)

    @staticmethod
    def is_pl_state_dict(maybe_pl_state_dict: dict[str, Any]) -> bool:
        """
        Returns whether given dict is a PL state dict/checkpoint.
        Based on a stupid heuristic.

        Parameters
        ----------
        maybe_pl_state_dict : dict[str, Any]
            The dict to check.

        Returns
        -------
        bool
            Whether it's a PL checkpoint.
        """
        # this is just a stupid heuristic
        return "epoch" in maybe_pl_state_dict and "state_dict" in maybe_pl_state_dict

    @staticmethod
    def _load_metrics(pl_state_dict) -> Metrics:
        state_dict = pl_state_dict["callbacks"]["MetricsTracker"]
        metrics = Metrics(state_dict)
        return metrics

    @staticmethod
    def _load_state_dict_from_path(path):
        import torch

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Tried loading last checkpoint but none found at: '{path}'"
            )

        state_dict = torch.load(path, map_location="cpu")

        if "pytorch-lightning_version" not in state_dict:
            warnings.warn(
                "Didn't load a PL checkpoint! This is not supposed to happen! "
                "The Experiment class is intended for development purposes!"
            )

        return state_dict


# ------------- Exceptions ----------------
class EMAWeightsNotFound(Exception):
    pass


class EmptyModelStateDictError(Exception):
    pass
