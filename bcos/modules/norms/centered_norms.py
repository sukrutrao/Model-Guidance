"""
Centered Norms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from bcos.modules.common import DynamicMultiplication


__all__ = [
    "AllNorm2d",
    "BatchNorm2d",
    "DetachableGroupNorm2d",
    "DetachableGNInstanceNorm2d",
    "DetachableGNLayerNorm2d",
    "DetachableLayerNorm",
]


# The easiest way is to use BN3D
class AllNorm2d(nn.BatchNorm3d):
    """
    The AllNorm.
    """

    def __init__(
        self,
        num_features: int,
        *args,
        **kwargs,
    ) -> None:
        # since we do it over the whole thing we have to set
        # this to one
        super().__init__(
            1,
            *args,
            **kwargs,
        )

    def forward(self, input: "Tensor") -> "Tensor":
        original_shape = input.shape
        # (B,C,H,W) -> (B,1,C,H,W)
        input = input.unsqueeze(1)

        # (B,1,C,H,W) normed
        output = super().forward(input)

        # (B,C,H,W) normed
        return output.reshape(original_shape)

    def set_explanation_mode(self, on: bool = True):
        if on:
            assert (
                not self.training
            ), "Centered AllNorm only supports explanation mode during .eval()!"


# just for a warnable version
class BatchNorm2d(nn.BatchNorm2d):
    def set_explanation_mode(self, on: bool = True):
        if on:
            assert (
                not self.training
            ), "Centered BN only supports explanation mode during .eval()!"


class DetachableGroupNorm2d(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_multiplication = DynamicMultiplication()

    def forward(self, input: "Tensor") -> "Tensor":
        # input validation
        assert input.dim() == 4, f"Expected 4D input got {input.dim()}D instead!"
        assert input.shape[1] % self.num_groups == 0, (
            "Number of channels in input should be divisible by num_groups, "
            f"but got input of shape {input.shape} and num_groups={self.num_groups}"
        )

        # ------------ manual GN forward pass -------------
        # separate the groups
        # (N, C, *) -> (N, G, C // G, *)
        N, C = input.shape[:2]
        x = input.reshape(N, self.num_groups, C // self.num_groups, *input.shape[2:])

        # calc stats
        var, mean = torch.var_mean(
            x, dim=tuple(range(2, x.dim())), unbiased=False, keepdim=True
        )
        std = (var + self.eps).sqrt()

        # normalize
        x = self.dynamic_multiplication(input=x - mean, weight=1 / std)

        # reshape back
        x = x.reshape(input.shape)

        # affine transformation
        if self.weight is not None:
            x = self.weight[None, ..., None, None] * x

        if self.bias is not None:
            x = x + self.bias[None, ..., None, None]

        return x


class DetachableGNInstanceNorm2d(DetachableGroupNorm2d):
    def __init__(self, num_channels: int, *args, **kwargs):
        super().__init__(
            num_groups=num_channels,
            num_channels=num_channels,
            *args,
            **kwargs,
        )


class DetachableGNLayerNorm2d(DetachableGroupNorm2d):
    def __init__(self, num_channels: int, *args, **kwargs):
        super().__init__(
            num_groups=1,
            num_channels=num_channels,
            *args,
            **kwargs,
        )


class DetachableLayerNorm(nn.LayerNorm):
    """
    A non-CNN detachable Layer Norm.
    This is used for the transformers!
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_multiplication = DynamicMultiplication()

    def forward(self, x: "Tensor") -> "Tensor":
        # if not detaching -> just use normal pytorch forward pass
        if not self.dynamic_multiplication.is_in_explanation_mode:
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )

        # ------------ manual LN detached forward pass -------------
        d_num = len(self.normalized_shape)

        # calc stats
        var, mean = torch.var_mean(
            x, dim=tuple(range(-d_num, 0)), unbiased=False, keepdim=True
        )
        std = (var + self.eps).sqrt_()

        # normalize
        x = self.dynamic_multiplication(input=x - mean, weight=1 / std)

        # affine transformation
        if self.weight is not None:
            x = self.weight * x

        if self.bias is not None:
            x = x + self.bias

        return x

