import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_error_map(target, pred):
    """Return element-wise absolute error map.

    Args:
        target: Ground-truth tensor of shape `(B, C, H, W)`.
        pred: Predicted tensor of shape `(B, C, H, W)`.

    Returns:
        Tensor of shape `(B, C, H, W)` containing `abs(target - pred)`.
    """
    error = abs(target - pred)
    return error


class SSIM(nn.Module):
    """
    Structural Similarity (SSIM) metric.

    Implementation is adapted from the fastMRI SSIM loss formulation and
    returns one SSIM score per batch item.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        """Compute SSIM score for each sample in a batch.

        Args:
            X: Predicted image tensor of shape `(B, C, H, W)`.
            Y: Target image tensor of shape `(B, C, H, W)`.
            data_range: Per-sample dynamic range tensor of shape `(B,)`.

        Returns:
            Tensor of shape `(B,)` with SSIM values.
        """
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        S = rearrange(S, "b c h w -> b (c h w)")

        return S.mean(dim=1)


class PSNR(nn.Module):
    """Peak Signal-to-Noise Ratio (PSNR) metric.

    Returns one PSNR value per batch item.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        """Compute per-sample PSNR.

        Args:
            X: Predicted image tensor of shape `(B, C, H, W)`.
            Y: Target image tensor of shape `(B, C, H, W)`.
            data_range: Per-sample dynamic range tensor of shape `(B,)`.

        Returns:
            Tensor of shape `(B,)` with PSNR values in dB.
        """
        # Y is target
        err = rearrange((X - Y) ** 2, "b c h w -> b (c h w)")
        mse = torch.mean(err, dim=1)
        return 10 * torch.log10(data_range**2 / mse)


class NMSE(nn.Module):
    """Normalized Mean Squared Error (NMSE) metric.

    Returns one NMSE value per batch item.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, X, Y):
        """Compute per-sample NMSE.

        Args:
            X: Predicted image tensor of shape `(B, C, H, W)`.
            Y: Target image tensor of shape `(B, C, H, W)`.

        Returns:
            Tensor of shape `(B,)` with NMSE values.
        """
        # Y is target
        err = rearrange(Y - X, "b c h w -> b (c h w)")
        err = (err**2).sum(dim=-1)
        den = (rearrange(Y, "b c h w -> b (c h w)") ** 2).sum(dim=-1)
        return err / den
