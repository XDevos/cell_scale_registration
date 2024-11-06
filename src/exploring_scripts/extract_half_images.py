#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import numpy as np
from skimage import io

SIZE = 512
OUT_FOLD = f"OUT_half_imgs_{SIZE}"


def extract_half_npy(folder):
    # extract the corresponding image in npy
    name = "scan_001_DAPI_000_ROI_converted_decon_ch00_3Dmasks.npy"
    img_file = os.path.join(folder, name)
    img_3d = np.load(img_file)
    extracted_img = img_3d[:, :SIZE, :SIZE]
    os.makedirs(f"{OUT_FOLD}", exist_ok=True)
    np.save(f"{OUT_FOLD}/{name}", extracted_img)
    print(f"img save at: {OUT_FOLD}/{name}")


def extract_half_img(folder, cycle="DAPI", ch="00"):
    # extract the corresponding image in tif
    name = f"scan_001_{cycle}_000_ROI_converted_decon_ch{ch}.tif"
    img_file = os.path.join(folder, name)
    img_3d = io.imread(img_file)
    extracted_img = img_3d[:, :SIZE, :SIZE]
    os.makedirs(f"{OUT_FOLD}", exist_ok=True)
    io.imsave(f"{OUT_FOLD}/{name}", extracted_img)
    print(f"img save at: {OUT_FOLD}/{name}")


def main():
    folder = "/home/xdevos/grey/users/xdevos/test_cell_scale_registration/roi_000"
    extract_half_npy(folder)
    extract_half_img(folder, "DAPI", "00")
    extract_half_img(folder, "DAPI", "01")
    cycle_list = [
        f"RT{i}"
        for i in [
            1,
            2,
            3,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            19,
            20,
            21,
            23,
            24,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        ]
    ]
    for cycle in cycle_list:
        extract_half_img(folder, cycle, "00")


if __name__ == "__main__":
    begin_time = datetime.now()
    main()
    print("\n==================== Normal termination ====================\n")
    print(f"Elapsed time: {datetime.now() - begin_time}")
