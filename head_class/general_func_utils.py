import fastmri
import torch
from utils.transform_util import add_gaussian_noise


class GaussianNoise:
    """
    add gaussian noise with scale coef to img, img should have scale [-1, 1]

    Args:
    ----------
    coef : float tensor
        scale coefficient for gaussian noise
    coef_range : torch tensor
        scale coefficient for gaussian noise with [min, max]
    """

    def __init__(self, coef: torch.tensor, coef_range: torch.tensor):
        self.coef = coef
        self.coef_range = coef_range #[min, max]

    def __call__(self, image):
        image_noisy = add_gaussian_noise(image, coef=self.coef)
        return image_noisy

    def _get_param(self):
        return self.coef

    def _get_norm_param(self):
        return self.coef

    def _get_param_range(self):
        return self.coef_range.unsqueeze(0)

    def _set_param(self, new_coef: torch.tensor):
        self.coef = new_coef


class ComplexAbs:
    """
    return absolute value [1, H, W] of a given tensor with structure [2, H, W]
    input last dim contains (real, imaginary) value

    Args:
    ----------
    image : tensor with structure [2, H, W]
    """

    def __call__(self, image):
        image = image.permute(1, 2, 0)  # [H, W, 2]
        image_abs = fastmri.complex_abs(image)  # [H, W]
        return image_abs.unsqueeze(0)
