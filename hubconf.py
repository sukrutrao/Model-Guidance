from typing import (
    Any as _Any,
    Dict as _Dict,
)

import torch
from bcos.experiments.utils import Experiment as _Experiment

dependencies = ["torch", "torchvision"]

BASE = "weights/bcos"

# map (base -> (model_name -> url))
URLS: _Dict[str, _Dict[str, str]] = {
    "bcos_final": {
        "resnet_50": f"{BASE}/resnet_50-f46c1a4159.pth",
    },
}


def _get_model(
    experiment_name: str,
    pretrained: bool,
    progress: bool,
    base_network: str = "bcos_final",
    dataset: str = "ImageNet",
    **model_kwargs: _Any
) -> torch.nn.Module:
    """
    Helper that loads the model and attaches its config and
    transform to it as `config` and `transform` respectively.
    """
    # load empty model
    exp = _Experiment(dataset, base_network, experiment_name)
    model = exp.get_model(**model_kwargs)

    # attach stuff
    model.config = exp.config
    model.transform = model.config["data"]["test_transform"]

    # load weights if needed
    if pretrained:
        url = URLS[base_network][experiment_name]
        state_dict = torch.load(url)
        model.load_state_dict(state_dict)

    return model


# ------------------------------- [model entrypoints] -------------------------------------
# TODO: add more info
def resnet18(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNet-18"""
    return _get_model("resnet_18", pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNe-34"""
    return _get_model("resnet_34", pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNet-50"""
    return _get_model("resnet_50", pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNet-101"""
    return _get_model("resnet_101", pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNet-152"""
    return _get_model("resnet_152", pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNeXt-50 32x4d"""
    return _get_model("resnext50_32x4d", pretrained, progress, **kwargs)


def densenet121(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos DenseNet-121"""
    return _get_model("densenet_121", pretrained, progress, **kwargs)


def densenet161(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos DenseNet-161"""
    return _get_model("densenet_161", pretrained, progress, **kwargs)


def densenet169(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos DenseNet-169"""
    return _get_model("densenet_169", pretrained, progress, **kwargs)


def densenet201(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos DenseNet-201"""
    return _get_model("densenet_201", pretrained, progress, **kwargs)


def vgg11_bnu(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos VGG-11 BNU (BN without centering)"""
    return _get_model("vgg_11_bnu", pretrained, progress, **kwargs)
