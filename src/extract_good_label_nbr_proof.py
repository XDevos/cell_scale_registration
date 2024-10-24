#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import numpy as np
from skimage.measure import regionprops
from skimage import io


def extract_nucleus_scale(folder, isonuc_bbox, cycle="DAPI", ch="00"):
    # extract the corresponding image in tif
    img_file = os.path.join(
        folder, f"scan_001_{cycle}_005_ROI_converted_decon_ch{ch}.tif"
    )
    img_3d = io.imread(img_file)
    zmin, ymin, xmin, zmax, ymax, xmax = isonuc_bbox
    extracted_img = img_3d[zmin:zmax, ymin:ymax, xmin:xmax]
    os.makedirs("OUT", exist_ok=True)
    io.imsave(f"OUT/extract_{cycle}_ch{ch}.tif", extracted_img)
    print(f"img save at: OUT/extract_{cycle}_ch{ch}.tif")


def main():
    folder = "/home/xdevos/grey/ProcessedData_2024/Experiment_49_David_RAMM_DNAFISH_Bantignies_proto_G1E_cells_LRKit/deinterleave_deconvolved_test/005"
    all_masks_file = os.path.join(
        folder, "mask_3d/data", "scan_001_DAPI_005_ROI_converted_decon_ch00_3Dmasks.npy"
    )
    label_img = np.load(all_masks_file)
    regions = regionprops(label_img)
    isolated_nucleus = regions[923]  # 924-1
    isonuc_center = isolated_nucleus.centroid
    print(f"center: {isonuc_center}")
    isonuc_bbox = isolated_nucleus.bbox
    print(f"bbox: {isonuc_bbox}")
    extract_nucleus_scale(folder, isonuc_bbox, "DAPI", "00")
    extract_nucleus_scale(folder, isonuc_bbox, "DAPI", "01")
    extract_nucleus_scale(folder, isonuc_bbox, "RT17", "00")
    extract_nucleus_scale(folder, isonuc_bbox, "RT17", "01")


if __name__ == "__main__":
    begin_time = datetime.now()
    main()
    print("\n==================== Normal termination ====================\n")
    print(f"Elapsed time: {datetime.now() - begin_time}")
