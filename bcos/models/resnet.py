"""
Modified from
https://github.com/chenyaofo/pytorch-cifar-models/blob/e9482ebc665084761ad9c84d36c83cbb82a164a9/pytorch_cifar_models/resnet.py

ResNeXt implementation based on https://github.com/pytorch/vision/blob/v0.13.1/torchvision/models/resnet.py

---
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/resnet.py

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from typing import Any, List, Optional, Callable, Type, Union

import torch.nn as nn
from torchvision.ops import StochasticDepth

from .bcos_common import BcosModelBase


__all__ = ["BcosResNet"]


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    conv_layer: Callable[..., nn.Module] = None,
):
    """3x3 convolution with padding"""
    return conv_layer(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    conv_layer: Callable[..., nn.Module] = None,
):
    """1x1 convolution"""
    return conv_layer(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Callable[..., nn.Module] = None,
        # act_layer: Callable[..., nn.Module] = None,
        stochastic_depth_prob: float = 0.0,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(
            inplanes,
            planes,
            stride,
            conv_layer=conv_layer,
        )
        self.bn1 = norm_layer(planes)
        # self.act = act_layer(inplace=True)
        self.conv2 = conv3x3(
            planes,
            planes,
            conv_layer=conv_layer,
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.stochastic_depth = (
            StochasticDepth(stochastic_depth_prob, "row")
            if stochastic_depth_prob
            else None
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stochastic_depth:
            out = self.stochastic_depth(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.act(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Callable[..., nn.Module] = None,
        # act_layer: Callable[..., nn.Module] = None,
        stochastic_depth_prob: float = 0.0,
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(
            inplanes,
            width,
            conv_layer=conv_layer,
        )
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(
            width,
            width,
            stride,
            groups,
            dilation,
            conv_layer=conv_layer,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(
            width,
            planes * self.expansion,
            conv_layer=conv_layer,
        )
        self.bn3 = norm_layer(planes * self.expansion)
        # self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.stochastic_depth = (
            StochasticDepth(stochastic_depth_prob, "row")
            if stochastic_depth_prob
            else None
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stochastic_depth:
            out = self.stochastic_depth(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.act(out)

        return out


class BcosResNet(BcosModelBase):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        in_chans: int = 6,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Callable[..., nn.Module] = None,
        # act_layer: Callable[..., nn.Module] = None,
        inplanes: int = 64,
        small_inputs: bool = False,
        stochastic_depth_prob: float = 0.0,
        **kwargs: Any,  # ignore rest
    ):
        super().__init__()

        if kwargs:
            print("The following args passed to model will be ignored", kwargs)

        if norm_layer is None:
            norm_layer = nn.Identity
        self._norm_layer = norm_layer
        self._conv_layer = conv_layer
        # self._act_layer = act_layer

        self.inplanes = inplanes
        self.dilation = 1
        n = len(layers)  # number of stages
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False] * (n - 1)
        if len(replace_stride_with_dilation) != n - 1:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a {n - 1}-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        if small_inputs:
            self.conv1 = conv3x3(
                in_chans,
                self.inplanes,
                conv_layer=conv_layer,
            )
            self.pool = None
        else:
            self.conv1 = conv_layer(
                in_chans,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
            )
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = norm_layer(self.inplanes)
        # self.act = act_layer(inplace=True)

        self.__total_num_blocks = sum(layers)
        self.__num_blocks = 0
        self.layer1 = self._make_layer(
            block,
            inplanes,
            layers[0],
            stochastic_depth_prob=stochastic_depth_prob,
        )
        self.layer2 = self._make_layer(
            block,
            inplanes * 2,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            stochastic_depth_prob=stochastic_depth_prob,
        )
        self.layer3 = self._make_layer(
            block,
            inplanes * 4,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            stochastic_depth_prob=stochastic_depth_prob,
        )
        try:
            self.layer4 = self._make_layer(
                block,
                inplanes * 8,
                layers[3],
                stride=2,
                dilate=replace_stride_with_dilation[2],
                stochastic_depth_prob=stochastic_depth_prob,
            )
            last_ch = inplanes * 8
        except IndexError:
            self.layer4 = None
            last_ch = inplanes * 4

        self.num_features = last_ch * block.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_classes = num_classes
        self.fc = conv_layer(
            self.num_features,
            self.num_classes,
            kernel_size=1,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        stochastic_depth_prob: float = 0.0,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        conv_layer = self._conv_layer
        # act_layer = self._act_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes,
                    planes * block.expansion,
                    stride,
                    conv_layer=conv_layer,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer=norm_layer,
                conv_layer=conv_layer,
                # act_layer=act_layer,
                stochastic_depth_prob=stochastic_depth_prob
                * self.__num_blocks
                / (self.__total_num_blocks - 1),
            )
        )
        self.__num_blocks += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    conv_layer=conv_layer,
                    # act_layer=act_layer,
                    stochastic_depth_prob=stochastic_depth_prob
                    * self.__num_blocks
                    / (self.__total_num_blocks - 1),
                )
            )
            self.__num_blocks += 1

        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.act(x)
        if self.pool is not None:
            x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        return x

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.fc = self._conv_layer(
            self.num_features,
            num_classes,
        )

    def forward(self, x):
        x = self.forward_features(x)

        x = self.fc(x)
        x = self.avgpool(x)
        x = x.flatten(1)

        return x


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool = False,
    progress: bool = True,
    inplanes: int = 64,
    **kwargs: Any,
) -> BcosResNet:
    model = BcosResNet(block, layers, inplanes=inplanes, **kwargs)
    if pretrained:
        raise ValueError(f"No pre-trained weights available!")
    return model


# ---------------------
# ResNets for ImageNet
# ---------------------
def resnet18(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def resnext50_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext101_32x8d(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs,
    )


def wide_resnet50_2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_resnet101_2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs,
    )


# ---------------------
# ResNets for CIFAR-10
# ---------------------
def _update_if_not_present(key, value, d):
    if key not in d:
        d[key] = value


def _update_default_cifar(kwargs) -> None:
    _update_if_not_present("num_classes", 10, kwargs)
    _update_if_not_present("small_inputs", True, kwargs)


def cifar10_resnet20(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnet20",
        BasicBlock,
        [3] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnet32(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnet32",
        BasicBlock,
        [5] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnet44(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnet44",
        BasicBlock,
        [7] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnet56(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnet56",
        BasicBlock,
        [9] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnet110(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnet110",
        BasicBlock,
        [18] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


# --------------------
# ResNeXts
# --------------------
# These are model configs specified in the ResNeXt Paper https://arxiv.org/pdf/1611.05431.pdf


def cifar10_resnext29_8x64d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 8
    kwargs["width_per_group"] = 64
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext29_8x64d",
        Bottleneck,
        [3] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=64,
        **kwargs,
    )


def cifar10_resnext29_16x64d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 16
    kwargs["width_per_group"] = 64
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext29_16x64d",
        Bottleneck,
        [3] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=64,
        **kwargs,
    )


# The model configs specified in the ResNeXt Paper
# are very large. We use smaller ones here.
# So instead of the 8x64d or 16x64d settings (with [64, 128, 256] widths)
# we have 32x4d and 16x8d settings ([16, 32, 64]).
# first 32x4d


def cifar10_resnext29_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext29_32x4d",
        Bottleneck,
        [3] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnext47_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext47_32x4d",
        Bottleneck,
        [5] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnext65_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext65_32x4d",
        Bottleneck,
        [7] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnext101_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext101_32x4d",
        Bottleneck,
        [11] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


# next with 16x8d


def cifar10_resnext29_16x8d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 16
    kwargs["width_per_group"] = 8
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext29_32x4d",
        Bottleneck,
        [3] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnext47_16x8d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 16
    kwargs["width_per_group"] = 8
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext47_32x4d",
        Bottleneck,
        [5] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnext65_16x8d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 16
    kwargs["width_per_group"] = 8
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext65_32x4d",
        Bottleneck,
        [7] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnext101_16x8d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 16
    kwargs["width_per_group"] = 8
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext101_32x4d",
        Bottleneck,
        [11] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )
