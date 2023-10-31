import warnings
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .common import DynamicMultiplication

__all__ = ["NormedConv2d", "BcosConv2d", "BcosConv2dWithScale"]


class NormedConv2d(nn.Conv2d):
    """
    Standard 2D convolution, but with unit norm weights.
    """

    def forward(self, in_tensor: Tensor) -> Tensor:
        w = self.weight / self.weight.norm(p=2, dim=(1, 2, 3), keepdim=True)
        return self._conv_forward(input=in_tensor, weight=w, bias=self.bias)


class BcosConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 1,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        # special (note no scale here! See BcosConv2dWithScale below)
        b: Union[int, float] = 2,
        max_out: int = 1,
        normalize_weights: bool = True,
        **kwargs,  # bias is always False
    ):
        assert max_out > 0, f"max_out should be greater than 0, was {max_out}"
        super().__init__()

        # save everything
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        self.bias = False

        self.b = b
        self.max_out = max_out
        self.normalized_weights = normalize_weights

        # check dilation
        if dilation > 1:
            warnings.warn("dilation > 1 is much slower!")
            self.calc_patch_norms = self._calc_patch_norms_slow

        # create base conv class
        if normalize_weights:
            conv_cls = NormedConv2d
        else:
            conv_cls = nn.Conv2d
        self.linear = conv_cls(
            in_channels=in_channels,
            out_channels=out_channels * max_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        # for multiplying with dynamically calculated weights
        # imp for getting explanations
        self.dynamic_multiplication = DynamicMultiplication()

    def forward(self, in_tensor: Tensor) -> Tensor:
        """
        Forward pass.
        Args:
            in_tensor: Input tensor. Expected shape: (B, C, H, W)

        Returns:
            BcosConv2d output on the input tensor.
        """
        # Simple linear layer
        out = self.linear(in_tensor)

        # MaxOut computation
        if self.max_out > 1:
            M = self.max_out
            C = self.out_channels
            out = out.unflatten(dim=1, sizes=(C, M))
            out = out.max(dim=2, keepdim=False).values

        # if B=1, no further calculation necessary
        if self.b == 1:
            return out

        # Calculating the norm of input patches. Use average pooling and upscale by kernel size.
        norm = self.calc_patch_norms(in_tensor)

        # add norm of weights if weights are not normalized
        if not self.normalized_weights:
            w = self.linear.weight
            norm = norm * w.norm(p=2, dim=(1, 2, 3))[..., None, None]

        # b = 2 => no need to explicitly calculate cos
        if self.b == 2:
            dynamic_weights = out.abs() / norm
        else:
            abs_cos = (out / norm).abs() + 1e-6  # |cos| term
            dynamic_weights = abs_cos.pow(self.b - 1)

        out = self.dynamic_multiplication(weight=dynamic_weights, input=out)
        return out

    def calc_patch_norms(self, in_tensor: Tensor) -> Tensor:
        """
        Calculates the norms of the patches.
        """
        squares = in_tensor**2
        if self.groups == 1:
            # normal conv
            squares = squares.sum(1, keepdim=True)
        else:
            G = self.groups
            C = self.in_channels
            # group channels together and sum reduce over them
            # ie [N,C,H,W] -> [N,G,C//G,H,W] -> [N,G,H,W]
            # note groups MUST come first
            squares = squares.unflatten(1, (G, C // G)).sum(2)

        norms = (
            F.avg_pool2d(
                squares,
                self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                divisor_override=1,
            )
            + 1e-6  # stabilizing term
        ).sqrt()

        if self.groups > 1:
            # norms.shape will be [N,G,H,W] (here H,W are spatial dims of output)
            # we need to convert this into [N,O,H,W] so that we can divide by this norm
            # (because we can't easily do broadcasting)
            N, G, H, W = norms.shape
            O = self.out_channels
            norms = (
                norms.unflatten(
                    1, (G, 1)
                )  # insert axis/dim between N and G; [N,G,H,W] -> [N,G,1,H,W]
                .expand(
                    N, G, O // G, H, W
                )  # expand this dim by #channels per output group
                .flatten(
                    1, 2
                )  # flatten the channel dims s.t. [N,G,O//G,H,W] -> [N,O,H,W]
            )

        return norms

    def _calc_patch_norms_slow(self, in_tensor: Tensor) -> Tensor:
        # this is much slower but definitely correct
        # use for testing or something difficult to implement
        # like dilation
        ones_kernel = torch.ones_like(self.linear.weight)

        return (
            F.conv2d(
                in_tensor**2,
                ones_kernel,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            + 1e-6
        ).sqrt()

    def extra_repr(self) -> str:
        # rest in self.linear
        s = "B={b}"

        if self.max_out > 1:
            s += ", max_out={max_out}"

        # final comma as self.linear is shown in next line
        s += ","

        return s.format(**self.__dict__)


class BcosConv2dWithScale(BcosConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 1,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        # special
        b: Union[int, float] = 2,
        max_out: int = 1,
        normalize_weights: bool = True,
        scale: Optional[float] = None,  # use default init
        scale_factor: Union[int, float] = 100.0,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            padding_mode,
            device,
            dtype,
            b,
            max_out,
            normalize_weights,
            **kwargs,
        )

        if scale is None:
            ks_scale = (
                kernel_size
                if not isinstance(kernel_size, tuple)
                else np.sqrt(np.prod(kernel_size))
            )
            self.scale = (ks_scale * np.sqrt(self.in_channels)) / scale_factor
        else:
            assert scale != 1.0, "For scale=1.0, use the normal BcosConv2d instead!"
            self.scale = scale

    def forward(self, in_tensor: Tensor) -> Tensor:
        out = super().forward(in_tensor)
        return out / self.scale

    def extra_repr(self) -> str:
        result = super().extra_repr()
        result = f"scale={self.scale:.3f}, " + result
        return result
