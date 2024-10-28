#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from data_manager import load_npy, save_to_compare
from compare import print_ssim_mse
from skimage.transform import warp_polar, SimilarityTransform, warp
from skimage.registration import phase_cross_correlation
from skimage.filters import window, difference_of_gaussians
from scipy.fft import fft2, fftshift


def rescale_target(target, shift_scale):
    tform = SimilarityTransform(scale=shift_scale)
    return warp(target, tform.inverse)


def shift_polar_simple(ref_2d, target_2d, higkh):
    # First, band-pass filter both images
    low = 2
    high = 5
    ref_gaus = difference_of_gaussians(ref_2d, low, high)
    targ_gaus = difference_of_gaussians(target_2d, low, high)

    # window images
    wimage = ref_gaus * window("hann", ref_gaus.shape)
    rts_wimage = targ_gaus * window("hann", targ_gaus.shape)

    # work with shifted FFT magnitudes
    image_fs = np.abs(fftshift(fft2(wimage)))
    rts_fs = np.abs(fftshift(fft2(rts_wimage)))

    # Create log-polar transformed FFT mag images and register
    shape = image_fs.shape
    radius = shape[0] // 8  # only take lower frequencies
    warped_image_fs = warp_polar(
        image_fs, radius=radius, output_shape=shape, scaling="log", order=0
    )
    warped_rts_fs = warp_polar(
        rts_fs, radius=radius, output_shape=shape, scaling="log", order=0
    )

    warped_image_fs = warped_image_fs[: shape[0] // 2, :]  # only use half of FFT
    warped_rts_fs = warped_rts_fs[: shape[0] // 2, :]
    shifts, error, phasediff = phase_cross_correlation(
        warped_image_fs, warped_rts_fs, upsample_factor=10
    )

    # Use translation parameters to calculate rotation and scaling parameters
    shiftr, shiftc = shifts[:2]
    klog = shape[1] / np.log(radius)
    shift_scale = 1 / np.exp(shiftc / klog)

    rescaled_target = rescale_target(target_2d, shift_scale)
    print(f"SHIFT_SCALE == {shift_scale}")
    return rescaled_target


def main():
    folder = "/home/xdevos/Repositories/XDevos/explore_registration/sample_924/"
    in_sub_folder = "3_global_shift"
    for funct in [
        "mip_projection",
        # "sum_projection",
        # "mean_projection",
        # "std_projection",
    ]:
        for low in range(1):
            ref_name = os.path.join(folder, in_sub_folder, funct, "ref.npy")
            target_name = os.path.join(folder, in_sub_folder, funct, "target.npy")
            ref_2d = load_npy(ref_name)
            target_2d = load_npy(target_name)
            scaled_targ = shift_polar_simple(ref_2d, target_2d, low)
            save_to_compare(
                ref_2d, scaled_targ, f"7_global_translate_polar_log/{funct}"
            )
            print_ssim_mse(ref_2d, scaled_targ)


if __name__ == "__main__":
    main()
