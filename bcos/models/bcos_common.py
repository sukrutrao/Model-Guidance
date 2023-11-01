"""
Common classes and mixins for B-cos models.
"""
from typing import Union, List, Optional, Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor

__all__ = ["BcosModelBase", "ExplanationModeContextManager", "BcosSequential"]


class ExplanationModeContextManager:
    """
    A context manager which activates and puts model in to explanation
    mode and deactivates it afterwards
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.expl_modules = None

    def find_expl_modules(self):
        self.expl_modules = [
            m for m in self.model.modules() if hasattr(m, "set_explanation_mode")
        ]

    def __enter__(self):
        if self.expl_modules is None:
            self.find_expl_modules()

        for m in self.expl_modules:
            m.set_explanation_mode(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m in self.expl_modules:
            m.set_explanation_mode(False)


class BcosModelBase(torch.nn.Module):
    """
    This base class defines useful explanation generation helpers.
    """

    to_probabilities = torch.sigmoid
    """ Function to convert outputs to probabilties. """

    def __init__(self):
        super().__init__()

        self._expl_mode_ctx = ExplanationModeContextManager(self)
        """ Context manager for explanation mode. """

    def explanation_mode(self):
        """
        Creates a context manager which puts model in to explanation
        mode and when exiting puts it into normal mode back again.
        """
        return self._expl_mode_ctx

    # TODO: end-user simple explanation generation method
    # FIXME: maybe allow unbatched input too?
    def forward_and_explain(
        self,
        in_tensor: "Tensor",
        idx: "Optional[Union[List[int], LongTensor]]" = None,
        color_explanations: bool = True,
        keep_graph: bool = False,
        **kwargs: Any,
    ) -> "Dict[str, Tensor]":
        """
        Get explanation/contribution/linear maps for model output for given input.
        Returns the above inside a dict.

        Parameters
        ----------
        in_tensor : Tensor
            The input for which to explain the corresponding outputs of
            the model. Input must be a 4D batched image tensor.
            Shape [N, C, H, W]
        idx : Optional[List[int] | LongTensor]
            What outputs to explain. If given, it should be a list/tensor
            representing class indices of the outputs to be explained.
            Shape [N,]
        color_explanations : bool
            Whether to return color explanations. Default: `True`
            This requires `in_tensor` to have 6 channel color input.
            [R, G, B, 1 - R, 1 - G, 1 - B]
        keep_graph: bool
            Whether to create and retain graph with backward passes.
            Default: `False`
        **kwargs : Any
            Optional keyword arguments to pass to `gradient_to_image`.
            This is only done if `color_explanations` is set to `True`.

        Returns
        -------
        dict[str, Tensor]
            A dictionary containing:
            "weight": the dynamically calculated weights. Shape [N, C, H, W]
            "output": The model output. Shape [N, K]
            "idx": The indices used. Argmax if predictions explained. Shape [N,]
            "contribution": Depending upon `color_explanations` either simply
                w(x) * x with shape [N, C, H, W] or decoded color explanations
                of shape [N, H, W, 4]. The former a `torch.Tensor` and the latter
                a `np.ndarray`
        """
        # some input validation
        assert (
            in_tensor.dim() == 4
        ), f"Only 4D (batched) images accepted! Got {in_tensor.dim()}D"
        if idx is not None:
            assert isinstance(
                idx, (list, Tensor)
            ), f"idx should be either a list/tensor of indices or None! but was {idx}"
            assert in_tensor.shape[0] == len(
                idx
            ), f"Batch sizes of input ({in_tensor.shape[0]}) and idx ({len(idx)} are different!"

        # ==== Actual Contribution/Linear Map Calculations ====
        in_tensor.grad = None
        in_tensor.requires_grad = True
        with self.explanation_mode():
            output = self(in_tensor)  # doesn't actually need to be in expl mode

            if idx is not None:
                logits_to_explain = output[range(len(in_tensor)), idx]
            else:  # explain prediction
                logits_to_explain, idx = output.max(dim=1)

            logits_to_explain.sum().backward(
                inputs=[in_tensor], create_graph=keep_graph, retain_graph=keep_graph
            )

        result = {
            "weight": in_tensor.grad,
            "output": output,
            "idx": idx,  # argmax if pred
        }
        if color_explanations:
            N, C, H, W = in_tensor.shape
            contributions = np.empty((N, H, W, 4))
            for b in range(N):
                contributions[b] = self.gradient_to_image(
                    image=in_tensor[b], linear_mapping=in_tensor.grad[b], **kwargs
                )
            result["contribution"] = contributions
        else:
            result["contribution"] = in_tensor.grad

        return result

    @classmethod
    def gradient_to_image(
        cls,
        image: "Tensor",
        linear_mapping: "Tensor",
        smooth: int = 0,
        alpha_percentile: float = 99.5,
    ) -> "np.ndarray":
        """
        Computing color image from dynamic linear mapping of B-cos models.
        From https://github.com/moboehle/B-cos/blob/0023500ce/interpretability/utils.py#L41.

        Args:
            image: Tensor
                Original input image (encoded with 6 color channels)
                Shape: [C, H, W] with C=6
            linear_mapping: Tensor
                Linear mapping W_{1\rightarrow l} of the B-cos model
                Shape: [C, H, W] same as image
            smooth: int
                Kernel size for smoothing the alpha values
            alpha_percentile: float
                Cut-off percentile for the alpha value
        Returns:
            np.ndarray
                image explanation of the B-cos model.
                Shape [H, W, C] (C=4 ie RGBA)
        """
        # shape of img and linmap is [C, H, W], summing over first dimension gives the contribution map per location
        contribs = (image * linear_mapping).sum(0, keepdim=True)
        contribs = contribs[0]
        # Normalise each pixel vector (r, g, b, 1-r, 1-g, 1-b) s.t. max entry is 1, maintaining direction
        rgb_grad = linear_mapping / (
            linear_mapping.abs().max(0, keepdim=True)[0] + 1e-12
        )
        # clip off values below 0 (i.e., set negatively weighted channels to 0 weighting)
        rgb_grad = rgb_grad.clamp(0)
        # normalise s.t. each pair (e.g., r and 1-r) sums to 1 and only use resulting rgb values
        rgb_grad = cls._to_numpy(rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12))

        # Set alpha value to the strength (L2 norm) of each location's gradient
        alpha = linear_mapping.norm(p=2, dim=0, keepdim=True)
        # Only show positive contributions
        alpha = torch.where(contribs[None] < 0, torch.zeros_like(alpha) + 1e-12, alpha)
        if smooth:
            alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth - 1) // 2)
        alpha = cls._to_numpy(alpha)
        alpha = (alpha / np.percentile(alpha, alpha_percentile)).clip(0, 1)

        rgb_grad = np.concatenate([rgb_grad, alpha], axis=0)
        # Reshaping to [H, W, C]
        grad_image = rgb_grad.transpose((1, 2, 0))
        return grad_image

    @staticmethod
    def _to_numpy(tensor: "Union[torch.Tensor, np.ndarray]") -> "np.ndarray":
        if not isinstance(tensor, torch.Tensor):
            return tensor
        return tensor.detach().cpu().numpy()

    @classmethod
    def plot_contribution_map(
        cls,
        contribution_map,
        ax=None,
        vrange=None,
        vmin=None,
        vmax=None,
        hide_ticks=True,
        cmap="bwr",
        percentile=100,
    ):
        """
        Visualises a contribution map, i.e., a matrix assigning individual weights to each spatial location.
        As default, this shows a contribution map with the "bwr" colormap and chooses vmin and vmax so that the map
        ranges from (-max(abs(contribution_map), max(abs(contribution_map)).
        From: https://github.com/moboehle/B-cos/blob/0023500cea7b/interpretability/utils.py#L78

        Args:
            contribution_map: (H, W) matrix to visualise as contributions.
            ax: axis on which to plot. If None, a new figure is created.
            vrange: If None, the colormap ranges from -v to v, with v being the maximum absolute value in the map.
                If provided, it will range from -vrange to vrange, as long as either one of the boundaries is not
                overwritten by vmin or vmax.
            vmin: Manually overwrite the minimum value for the colormap range instead of using -vrange.
            vmax: Manually overwrite the maximum value for the colormap range instead of using vrange.
            hide_ticks: Sets the axis ticks to []
            cmap: colormap to use for the contribution map plot.
            percentile: If percentile is given, this will be used as a cut-off for the attribution maps.
        Returns:
            The axis on which the contribution map was plotted.
        """
        assert (
            contribution_map.ndim == 2
        ), "Contribution map is supposed to only have 2 spatial dimensions."
        contribution_map = cls._to_numpy(contribution_map)
        cutoff = np.percentile(np.abs(contribution_map), percentile)
        contribution_map = np.clip(contribution_map, -cutoff, cutoff)

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1)

        if vrange is None or vrange == "auto":
            vrange = np.max(np.abs(contribution_map.flatten()))
        im = ax.imshow(
            contribution_map,
            cmap=cmap,
            vmin=-vrange if vmin is None else vmin,
            vmax=vrange if vmax is None else vmax,
        )

        if hide_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        return ax, im

    def attribute(self, image, target, **kwargs):
        """
        This method returns the contribution map according to Input x Gradient.
        Specifically, if the prediction model is a dynamic linear network, it returns the contribution map according
        to the linear mapping (IxG with detached dynamic weights).
        Args:
            image: Input image.
            target: Target class to check contributions for.
            kwargs: just for compatibility...
        Returns: Contributions for desired level.

        """
        _ = kwargs
        from interpretability.explanation_methods.explainers.captum import IxG

        with self.explanation_mode():
            attribution_f = IxG(self)
            att = attribution_f.attribute(image, target)

        return att

    @torch.no_grad()
    def attribute_selection(self, image, targets, **kwargs):
        """
        Runs trainer.attribute for the list of targets.


        Args:
            image: Input image.
            targets: Target classes to check contributions for.
            kwargs: just for compatibility...

        Returns: Contributions for desired level.

        """
        _ = kwargs
        return torch.cat([self.attribute(image, t) for t in targets], dim=0)


class BcosSequential(BcosModelBase, torch.nn.Sequential):
    def __init__(self, *args):
        BcosModelBase.__init__(self)
        torch.nn.Sequential.__init__(self, *args)
