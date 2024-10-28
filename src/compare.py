#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


def print_ssim_mse(ref, targ):
    ssim_none = ssim(ref, targ)
    print(f"SSIM: {ssim_none}")
    mse_none = mean_squared_error(ref, targ)
    print(f"MSE: {mse_none}")
