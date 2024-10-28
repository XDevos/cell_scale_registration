#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from data_manager import load_npy, save_to_compare
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from compare import print_ssim_mse


def shift_target(ref, target):
    shift_values, _, _ = phase_cross_correlation(ref, target, upsample_factor=100)
    print(f"shift values: {shift_values}")
    return shift(target, shift_values)


def main():
    # folder = "/home/xdevos/Repositories/XDevos/explore_registration/sample_924/"
    # in_sub_folder = "2_project"
    # for funct in [
    #     "mip_projection",
    #     "sum_projection",
    #     "mean_projection",
    #     "std_projection",
    # ]:
    folder = "/home/xdevos/Repositories/XDevos/explore_registration/sample_924/"
    in_sub_folder = "7_global_translate_polar_log"
    for funct in ["mip_projection"]:
        ref_name = os.path.join(folder, in_sub_folder, funct, "ref.npy")
        target_name = os.path.join(folder, in_sub_folder, funct, "target.npy")
        ref_2d = load_npy(ref_name)
        target_2d = load_npy(target_name)
        shifted_targ = shift_target(ref_2d, target_2d)
        save_to_compare(ref_2d, shifted_targ, f"8_global_shift/{funct}")
        print_ssim_mse(ref_2d, shifted_targ)


if __name__ == "__main__":
    main()
