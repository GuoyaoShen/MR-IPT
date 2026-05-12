"""Loss functions for image denoising."""

from typing import Annotated, TypeAlias

import torch
from torch import nn
import torch.nn.functional as F

Image2D: TypeAlias = Annotated[torch.Tensor, "(b, c, h, w)"]
image3D: TypeAlias = Annotated[torch.Tensor, "(b, c, h, w, d)"]
ImageTensor: TypeAlias = Image2D | image3D


class Gradient:
    """Computes the spatial gradient magnitude of a torch Tensor."""

    def __call__(self, data: ImageTensor) -> torch.Tensor:
        """Compute the gradient magnitude using central differences.

        Args:
            data: Input data tensor.

        Returns:
            Gradient magnitude of the same shape as ``data``.
        """
        spatial_dims = list(range(2, data.ndim))
        grads = torch.gradient(data, dim=spatial_dims)
        return torch.sqrt(torch.stack([g**2 for g in grads]).sum(dim=0) + 1e-8)


class MixedL1GradientLoss(nn.Module):
    """L1 loss combined with a gradient-magnitude loss for sharpness preservation.

    Args:
        gradient_weight: Weight applied to the gradient loss term. The L1 term
            is always weighted 1.0, so values < 1.0 make gradient a secondary
            objective and values > 1.0 prioritize edge sharpness over pixel
            accuracy.

    Notes:
        Supported input shapes are ``(b, c, h, w)`` and
        ``(b, c, h, w, d)``. The first two dimensions are interpreted as
        batch and channel dimensions.
    """

    def __init__(self, gradient_weight: float = 0.1) -> None:
        """Initialize with the gradient loss weight."""
        super().__init__()
        self.gradient_weight = gradient_weight
        self._gradient = Gradient()

    def forward(self, prediction: ImageTensor, target: ImageTensor) -> torch.Tensor:
        """Compute the mixed L1 and gradient loss.

        Args:
            prediction: Predicted tensor with shape ``(b, c, h, w)`` or
                ``(b, c, h, w, d)``.
            target: Ground-truth tensor with the same shape as ``prediction``.

        Returns:
            Scalar loss tensor.
        """
        l1 = F.l1_loss(prediction, target)
        grad = F.l1_loss(self._gradient(prediction), self._gradient(target))
        return l1 + self.gradient_weight * grad
