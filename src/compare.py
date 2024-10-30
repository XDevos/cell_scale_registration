#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def normalize_img(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def print_ssim_mse(ref, targ):
    print("Normalize both images before comparaison...")
    norm_ref = normalize_img(ref)
    norm_target = normalize_img(targ)
    ssim_none = ssim(norm_ref, norm_target)
    print(f"SSIM: {ssim_none}")
    mse_none = mean_squared_error(norm_ref, norm_target)
    print(f"MSE: {mse_none}")
