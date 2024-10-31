#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage import measure
from data_manager import load_npy, load_tif, save_to_compare
from compare import print_ssim_mse
from skimage.transform import warp_polar, SimilarityTransform, warp

from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

def extract_mask924(ref_3d, target_3d):
    folder = "/home/xdevos/Repositories/repo-XDevos/cell_scale_registration/INPUT/"
    mask_file = "scan_001_DAPI_005_ROI_converted_decon_ch00_3d_registered_3Dmasks.npy"
    labeled_masks = load_npy(folder + mask_file)
    mask_924 = np.where(labeled_masks == 924, 1, 0)
    props = measure.regionprops(mask_924)
    if len(props) != 1:
        raise ValueError("There are different mask n°924")
    # masked_ref = np.where(mask_924 == 1, ref_3d, 0)
    # masked_target = np.where(mask_924 == 1, target_3d, 0)
    minz, minx, miny, maxz, maxx, maxy = props[0].bbox
    bbox_ref = ref_3d[minz:maxz, minx:maxx, miny:maxy]
    bbox_target = target_3d[minz:maxz, minx:maxx, miny:maxy]
    return bbox_ref, bbox_target

def extract_rt17_larger(ref_3d,rt17_3d):
    folder = "/home/xdevos/Repositories/repo-XDevos/cell_scale_registration/INPUT/"
    mask_file = "scan_001_DAPI_005_ROI_converted_decon_ch00_3d_registered_3Dmasks.npy"
    labeled_masks = load_npy(folder + mask_file)
    mask_924 = np.where(labeled_masks == 924, 1, 0)
    props = measure.regionprops(mask_924)
    if len(props) != 1:
        raise ValueError("There are different mask n°924")
    # masked_ref = np.where(mask_924 == 1, ref_3d, 0)
    # masked_target = np.where(mask_924 == 1, target_3d, 0)
    minz, minx, miny, maxz, maxx, maxy = props[0].bbox
    print(ref_3d.shape)
    print(props[0].bbox)
    x_size = 5
    y_size = 10
    bbox_ref = ref_3d[minz:maxz, minx:maxx, miny:maxy]
    bbox_target = rt17_3d[minz:maxz, minx-x_size:maxx+x_size, miny-y_size:maxy+y_size]
    return bbox_ref, bbox_target, x_size, y_size

# todo remove because duplicated with translatepolarlog
def zoom_on_center(target, shift_scale):
    shape = target.shape
    center = (shape[0] / 2, shape[1] / 2)  # Centre de l'image
    translation_center = (-center[0] * (shift_scale - 1), -center[1] * (shift_scale - 1))
    similarity_transform = SimilarityTransform(scale=shift_scale, translation= translation_center)
    return warp(target, similarity_transform, output_shape=shape, preserve_range=True)
    
def shift_target(ref, target):
    shift_values, _, _ = phase_cross_correlation(ref, target, upsample_factor=100)
    print(f"shift values: {shift_values}")
    return shift(target, shift_values)

def main():
    folder = "/home/xdevos/Repositories/repo-XDevos/cell_scale_registration/INPUT/"
    ref_name = "scan_001_DAPI_005_ROI_converted_decon_ch01.tif"
    target_name = "scan_001_RT17_005_ROI_converted_decon_ch00.tif"
    ref_3d = load_tif(folder + ref_name)
    target_3d = load_tif(folder + target_name)
    ref, targ = extract_mask924(ref_3d, target_3d)
    save_to_compare(ref, targ, "1_raw")
    print_ssim_mse(ref, targ)

    bbox_ref, larger_rt17, x_size, y_size = extract_rt17_larger(ref_3d, target_3d)
    shifted_larg = shift(larger_rt17, [0,2.17,7.03])
    #print(f"max = {np.max(shifted_larg)}")
    for z in range(len(shifted_larg)):
        shifted_larg[z] = zoom_on_center(shifted_larg[z], 0.8292502770175191)
        # print(f"max = {np.max(shifted_larg[z])}")
    local_shifted = shift(shifted_larg, [0,-3.21,1.05])
    # print(f"max = {np.max(local_shifted)}")
    targ_to_comp = local_shifted[:,x_size:-x_size,y_size:-y_size]
    # print(f"max = {np.max(targ_to_comp)}")
    # print(targ_to_comp)
    # print(bbox_ref.shape)
    cropped_ref = bbox_ref[:,10:-10,10:-10]
    cropped_targ = targ_to_comp[:,10:-10,10:-10]
    save_to_compare(cropped_ref, cropped_targ, "10_raw")
    print_ssim_mse(cropped_ref, cropped_targ)
    crop_shift = shift_target(cropped_ref,cropped_targ)
    save_to_compare(cropped_ref, crop_shift, "11_raw")
    print_ssim_mse(cropped_ref, crop_shift)


if __name__ == "__main__":
    main()
