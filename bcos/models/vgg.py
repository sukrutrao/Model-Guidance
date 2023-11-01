from typing import Union, List, Dict, Any, cast, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from bcos.models.bcos_common import BcosModelBase

__all__ = [
    "BcosVGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class BcosVGG(BcosModelBase):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        conv_layer: Callable[..., nn.Module] = None,
    ) -> None:
        super(BcosVGG, self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            conv_layer(512, 4096, kernel_size=7, padding=3, scale_fact=1000),
            # nn.ReLU(True),
            # nn.Dropout(),
            conv_layer(4096, 4096, scale_fact=1000),
            # nn.ReLU(True),
            # nn.Dropout(),
            conv_layer(4096, num_classes, scale_fact=1000),
        )
        self.num_classes = num_classes
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)[..., None, None]
        x = self.classifier(x)
        # because it's supposed to come after
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    def get_features(self, x):
        return self.features(x)

    def get_sequential_model(self):
        model = nn.Sequential(*[m for m in self.features], self.classifier)
        return model

    def get_layer_idx(self, idx):
        return int(np.ceil(len(self.get_sequential_model()) * idx / 10))

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_layers(
    cfg: List[Union[str, int]],
    norm_layer: Callable[..., nn.Module],
    conv_layer: Callable[..., nn.Module],
    in_channels: int = 6,
    no_pool=False,
) -> nn.Sequential:
    # not None requirement is just to avoid mistakes
    # rather use isinstance(..., nn.Identity) as a check over None which
    # is usually a default argument
    assert norm_layer is not None, "Provide a norm layer class!"
    assert conv_layer is not None, "Provide a conv layer class!"
    layers: List[nn.Module] = []
    new_config = []
    for idx, entry in enumerate(cfg):
        new_config.append([entry, 1])
        if entry == "M" and no_pool:
            new_config[idx - 1][1] = 2

    for v, stride in new_config:
        if v == "M":
            if no_pool:
                continue
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = conv_layer(
                in_channels, v, kernel_size=3, padding=1, stride=stride, scale_fact=1000
            )
            if not isinstance(norm_layer, nn.Identity):
                layers += [conv2d, norm_layer(v)]  # , nn.ReLU(inplace=True)]
            else:
                layers += [conv2d]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(
    arch: str,
    cfg: str,
    pretrained: bool,
    progress: bool,
    norm_layer: Callable[..., nn.Module] = None,
    conv_layer: Callable[..., nn.Module] = None,
    in_chans: int = 6,
    no_pool=False,
    **kwargs: Any
) -> BcosVGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = BcosVGG(
        make_layers(
            cfgs[cfg],
            norm_layer=norm_layer,
            conv_layer=conv_layer,
            in_channels=in_chans,
            no_pool=no_pool,
        ),
        conv_layer=conv_layer,
        **kwargs,
    )
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BcosVGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", pretrained, progress, norm_layer=nn.Identity, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BcosVGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    assert "norm_layer" in kwargs
    assert not isinstance(kwargs["norm_layer"], nn.Identity)
    return _vgg("vgg11_bn", "A", pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BcosVGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13", "B", pretrained, progress, norm_layer=nn.Identity, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BcosVGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    assert "norm_layer" in kwargs
    assert not isinstance(kwargs["norm_layer"], nn.Identity)
    return _vgg("vgg13_bn", "B", pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BcosVGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", pretrained, progress, norm_layer=nn.Identity, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BcosVGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    assert "norm_layer" in kwargs
    assert not isinstance(kwargs["norm_layer"], nn.Identity)
    return _vgg("vgg16_bn", "D", pretrained, progress, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BcosVGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19", "E", pretrained, progress, norm_layer=nn.Identity, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BcosVGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    assert "norm_layer" in kwargs
    assert not isinstance(kwargs["norm_layer"], nn.Identity)
    return _vgg("vgg19_bn", "E", pretrained, progress, **kwargs)
