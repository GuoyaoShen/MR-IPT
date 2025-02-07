import numpy as np
import matplotlib.pyplot as plt

import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import fastmri
from fastmri.data import subsample, transforms, mri_data
from help_func import print_var_detail


class DataTransform:
    def __init__(
        self,
        mask_func,
    ):
        self.mask_func = mask_func

    def __call__(self, image: torch.tensor):
        kspace = torch.view_as_real(torch.fft.fftshift(torch.fft.fft2(image[0])))  # [H, W, 2]
        kspace = kspace[None, ...] # [H,W,2] to [1,H,W,2]

        # ====== Under-sampled ======
        # apply mask
        if isinstance(self.mask_func, subsample.MaskFunc):
            masked_kspace, mask, _ = transforms.apply_mask(kspace, self.mask_func)  # mask [1,1,W,1]
            mask = mask.squeeze(-1).squeeze(0).repeat(kspace.shape[1], 1)  # [H,W]
        else:
            masked_kspace, mask = apply_mask(kspace, self.mask_func)  # mask [1,H,W,1]
            mask = mask.squeeze(-1).squeeze(0)  # [H,W]

        image_masked = torch.fft.ifft2(torch.view_as_complex(masked_kspace))
        return image_masked, image, mask.unsqueeze(0)

def apply_mask(data, mask_func):
    '''
    data: [Nc,H,W,2]
    mask_func: return [Nc(1),H,W]
    '''
    mask, _ = mask_func()
    mask = torch.from_numpy(mask)
    mask = mask[..., None]  # [Nc(1),H,W,1]
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros
    return masked_data, mask
