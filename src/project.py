#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from data_manager import load_tif, save_to_compare
from compare import print_ssim_mse


def mip_projection(img_3d):
    return img_3d.max(axis=0)


def sum_projection(img_3d):
    projected_image = np.zeros((img_3d.shape[1], img_3d.shape[2]))
    for z_plan in img_3d:
        projected_image += z_plan
    return projected_image


def mean_projection(img_3d):
    return img_3d.mean(axis=0)


def std_projection(img_3d):
    return img_3d.std(axis=0)


def main():
    folder = "/home/xdevos/Repositories/XDevos/explore_registration/sample_924/"
    sub_folder = "1_raw"
    ref_name = os.path.join(folder, sub_folder, "ref.tif")
    target_name = os.path.join(folder, sub_folder, "target.tif")
    ref_3d = load_tif(ref_name)
    target_3d = load_tif(target_name)
    for funct in [mip_projection, sum_projection, mean_projection, std_projection]:
        ref_2d = funct(ref_3d)
        targ_2d = funct(target_3d)
        save_to_compare(ref_2d, targ_2d, f"2_project/{funct.__name__}")
        print_ssim_mse(ref_2d, targ_2d)


if __name__ == "__main__":
    main()
