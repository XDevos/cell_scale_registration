#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage import measure
from data_manager import load_npy, load_tif, save_to_compare
from compare import print_ssim_mse


def extract_mask924(ref_3d, target_3d):
    folder = "/home/xdevos/grey/ProcessedData_2024/Experiment_49_David_RAMM_DNAFISH_Bantignies_proto_G1E_cells_LRKit/deinterleave_deconvolved_test/005/mask_3d/data/"
    mask_file = "scan_001_DAPI_005_ROI_converted_decon_ch00_3Dmasks.npy"
    labeled_masks = load_npy(folder + mask_file)
    mask_924 = np.where(labeled_masks == 924, 1, 0)
    props = measure.regionprops(mask_924)
    if len(props) != 1:
        raise ValueError("There are different mask n°924")
    masked_ref = np.where(mask_924 == 1, ref_3d, 0)
    masked_target = np.where(mask_924 == 1, target_3d, 0)
    minz, minx, miny, maxz, maxx, maxy = props[0].bbox
    bbox_ref = masked_ref[minz:maxz, minx:maxx, miny:maxy]
    bbox_target = masked_target[minz:maxz, minx:maxx, miny:maxy]
    return bbox_ref, bbox_target


def main():
    folder = "/home/xdevos/grey/ProcessedData_2024/Experiment_49_David_RAMM_DNAFISH_Bantignies_proto_G1E_cells_LRKit/deinterleave_deconvolved_test/005/"
    ref_name = "scan_001_DAPI_005_ROI_converted_decon_ch01.tif"
    target_name = "scan_001_RT17_005_ROI_converted_decon_ch00.tif"
    ref_3d = load_tif(folder + ref_name)
    target_3d = load_tif(folder + target_name)
    ref, targ = extract_mask924(ref_3d, target_3d)
    save_to_compare(ref, targ, "1_raw")
    print_ssim_mse(ref, targ)


if __name__ == "__main__":
    main()
