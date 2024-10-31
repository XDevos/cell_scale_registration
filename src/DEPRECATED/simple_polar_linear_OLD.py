#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from data_manager import load_npy, save_to_compare
from compare import print_ssim_mse
from skimage.transform import warp_polar, SimilarityTransform, warp
from skimage.registration import phase_cross_correlation


def rescale_target(target, shift_scale):
    tform = SimilarityTransform(scale=shift_scale)
    return warp(target, tform.inverse)


def shift_polar_simple(ref_2d, target_2d):
    radius = min(ref_2d.shape)
    ref_polar = warp_polar(ref_2d, radius=radius, scaling="linear")
    target_polar = warp_polar(target_2d, radius=radius, scaling="linear")
    # setting `upsample_factor` can increase precision
    shifts, _, _ = phase_cross_correlation(ref_polar, target_polar, upsample_factor=100)
    shift_rotation, shiftc = shifts[:2]
    # Calculate scale factor from LINEAR translation
    shift_scale = 1 + (shiftc / radius)
    rescaled_target = rescale_target(target_2d, shift_scale)
    return rescaled_target


def main():
    folder = "/home/xdevos/Repositories/XDevos/explore_registration/sample_924/"
    in_sub_folder = "3_global_shift"
    for funct in [
        "mip_projection",
        "sum_projection",
        "mean_projection",
        "std_projection",
    ]:
        ref_name = os.path.join(folder, in_sub_folder, funct, "ref.npy")
        target_name = os.path.join(folder, in_sub_folder, funct, "target.npy")
        ref_2d = load_npy(ref_name)
        target_2d = load_npy(target_name)
        scaled_targ = shift_polar_simple(ref_2d, target_2d)
        save_to_compare(ref_2d, scaled_targ, f"4_global_polar_linear/{funct}")
        print_ssim_mse(ref_2d, scaled_targ)


if __name__ == "__main__":
    main()
