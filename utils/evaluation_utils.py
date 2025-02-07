import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio as psnr_tensor
from torchmetrics.functional import structural_similarity_index_measure as ssim_tensor


def get_error_map(target, pred):
    error = abs(target - pred)
    return error


class SSIM(nn.Module):
    """
    SSIM module. From fastMRI SSIM loss
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
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
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
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        S = S.view(S.shape[0], S.shape[-2] * S.shape[-1])

        return S.mean(dim=1)


class PSNR(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        # Y is target
        err = ((X - Y) ** 2).reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
        mse = torch.mean(err, dim=1)
        return (10 * torch.log10(data_range ** 2 / mse))


class NMSE(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, X, Y):
        # Y is target
        err = (Y - X).reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
        err = (err ** 2).sum(dim=-1)
        den = (Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2] * Y.shape[3]) ** 2).sum(dim=-1)
        return err / den
