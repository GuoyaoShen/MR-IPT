import fastmri
import torch
from fastmri.data import transforms, subsample


class KspaceDownSample:
    """
    down sample a given image [2, H, W] in its kspace with acceleration and center_fractions

    Args:
    ----------
    acceleration : int
        acceleration coefficient for kspace mask
    center_fraction : float
        center_fractions coefficient for kspace mask
    """

    def __init__(self, acceleration: torch.tensor, center_fraction: torch.tensor, acceleration_range: torch.tensor,
                 center_fraction_range: torch.tensor):
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.acceleration_range = acceleration_range #[min, max]
        self.center_fraction_range = center_fraction_range #[min, max]
        self.mask_func = subsample.RandomMaskFunc(
            center_fractions=[self.center_fraction.item()],
            accelerations=[self.acceleration.item()]
        )
        self.mask = 0

    def __call__(self, image: torch.tensor):
        image = image.permute(1, 2, 0)  # [H, W, 2]

        kspace = fastmri.fft2c(image)

        # apply mask
        kspace_masked, mask, _ = transforms.apply_mask(kspace, self.mask_func)  # apply mask in kspace [H, W, 2]
        mask = mask.squeeze(2).repeat(kspace_masked.shape[0], 1)  # [H,W]
        self.mask = mask

        img_masked = fastmri.ifft2c(kspace_masked)  # [H, W, 2]
        return img_masked.permute(2, 0, 1)  # [2, H, W]

    def _get_param(self):
        return torch.cat((self.acceleration,self.center_fraction), dim = 0)

    def _get_norm_param(self):
        acceleration_param = (self.acceleration - 1) / self.acceleration  # [0-1]
        center_fractions_param = 1 - self.center_fraction  # [0-1]
        return torch.cat((acceleration_param, center_fractions_param), dim=0)

    def _get_param_range(self):
        return torch.stack((self.acceleration_range,self.center_fraction_range), dim = 0)

    def _set_param(self, param: torch.tensor):
        self.acceleration[0] = param[0]
        self.center_fraction[0] = param[1]

class KspaceDownSampleRad:
    """
    down sample a given image [1, H, W] in its kspace with acceleration and center_fractions

    Args:
    ----------
    acceleration : tensor int
        acceleration coefficient for kspace mask
    center_fraction : tensor float
        center_fractions coefficient for kspace mask
    acceleration_range : tensor [min, max]
        range of acceleration if randomize kspace samples
    center_fractions_range : tensor [min, max]
        range of center_fractions if randomize kspace samples
    """

    def __init__(self, acceleration: torch.tensor, center_fraction: torch.tensor, acceleration_range: torch.tensor,
                 center_fraction_range: torch.tensor):
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.acceleration_range = acceleration_range #[min, max]
        self.center_fraction_range = center_fraction_range #[min, max]
        self.mask_func = subsample.RandomMaskFunc(
            center_fractions=[self.center_fraction.item()],
            accelerations=[self.acceleration.item()]
        )
        self.mask = 0

    def __call__(self, image: torch.tensor):
        kspace = torch.view_as_real(torch.fft.fftshift(torch.fft.fft2(image[0]))) # [H, W, 2]

        # apply mask
        kspace_masked, mask, _ = transforms.apply_mask(kspace, self.mask_func)  # apply mask in kspace [H, W, 2]
        mask = mask.squeeze(2).repeat(kspace.shape[0], 1)  # [H,W]
        self.mask = mask

        img_masked = torch.fft.ifft2(torch.view_as_complex(kspace_masked))  # [H, W]
        return img_masked.unsqueeze(0)  # [1, H, W]

    def _get_param(self):
        return torch.cat((self.acceleration,self.center_fraction), dim = 0)

    def _get_norm_param(self):
        acceleration_param = (self.acceleration - 1) / self.acceleration  # [0-1]
        center_fractions_param = 1 - self.center_fraction  # [0-1]
        return torch.cat((acceleration_param, center_fractions_param), dim=0)

    def _get_param_range(self):
        return torch.stack((self.acceleration_range,self.center_fraction_range), dim = 0)

    def _set_param(self, param: torch.tensor):
        self.acceleration[0] = param[0]
        self.center_fraction[0] = param[1]

