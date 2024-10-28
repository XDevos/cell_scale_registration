#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import numpy as np
from skimage.measure import regionprops
from skimage import io

import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

# def clear_data_outside_radius(img_3d, radius):


def extract_nucleus_scale(folder, center, isonuc_bbox, cycle="DAPI", ch="00"):
    # extract the corresponding image in tif
    radius = (
        max(isonuc_bbox[4] - isonuc_bbox[1], isonuc_bbox[5] - isonuc_bbox[2]) // 2 + 1
    )
    img_file = os.path.join(
        folder, f"scan_001_{cycle}_005_ROI_converted_decon_ch{ch}.tif"
    )
    img_3d = io.imread(img_file)
    zmin, _, _, zmax, _, _ = isonuc_bbox
    ymin = int(center[1]) - radius
    ymax = int(center[1]) + radius
    xmin = int(center[2]) - radius
    xmax = int(center[2]) + radius
    extracted_img = img_3d[zmin:zmax, ymin:ymax, xmin:xmax]

    out_dir = "OUT/extract_arround_centroid/"
    os.makedirs(out_dir, exist_ok=True)
    io.imsave(f"{out_dir}extract_{cycle}_ch{ch}.tif", extracted_img)
    print(f"img save at: {out_dir}extract_{cycle}_ch{ch}.tif")


def extract_cell_scale_tif(folder, cycle_list, label_nbr):
    all_masks_file = os.path.join(
        folder, "mask_3d/data", "scan_001_DAPI_005_ROI_converted_decon_ch00_3Dmasks.npy"
    )
    label_img = np.load(all_masks_file)
    regions = regionprops(label_img)
    isolated_nucleus = regions[
        label_nbr - 1
    ]  # first index is 0 and in labeled masks, zero corresponding to the background
    isonuc_center = isolated_nucleus.centroid
    print(f"center: {isonuc_center}")
    isonuc_bbox = isolated_nucleus.bbox
    print(f"bbox: {isonuc_bbox}")
    for cycle in cycle_list:
        extract_nucleus_scale(folder, isonuc_center, isonuc_bbox, cycle, "00")
        extract_nucleus_scale(folder, isonuc_center, isonuc_bbox, cycle, "01")


def load_fiducial(in_dir, cycle):
    ch = "01" if cycle == "DAPI" else "00"
    return io.imread(f"{in_dir}extract_{cycle}_ch{ch}.tif")


def save_png(data, output_path):
    fig = plt.figure()
    size = (10, 10)
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    norm = ImageNormalize(stretch=SqrtStretch())
    ax.set_title("2D Data")
    fig.add_axes(ax)
    ax.imshow(data, origin="lower", cmap="Greys_r", norm=norm)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"> $OUTPUT{output_path}")


def extract_circle(image_np, radius_reduction=0.1):

    # Obtenir les dimensions de l'image
    height, width = image_np.shape
    center = [height / 2, width / 2]
    radius = min(height // 2, width // 2) * (1 - radius_reduction)
    # Créer un masque circulaire
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius

    # Appliquer le masque circulaire
    circular_image = np.zeros_like(image_np)
    circular_image[mask] = image_np[mask]

    return circular_image


def global_align(cycle_list):
    in_dir = "OUT/extract_arround_centroid/"
    out_dir = "OUT/global_registration/"
    os.makedirs(f"{out_dir}RAW/")
    os.makedirs(f"{out_dir}registered/")
    ref = cycle_list.pop(0)
    for cycle in cycle_list:
        ref_img = load_fiducial(in_dir, ref)
        target_img = load_fiducial(in_dir, cycle)
        ref_proj = extract_circle(ref_img.max(axis=0), radius_reduction=0)
        target_proj = extract_circle(target_img.max(axis=0))
        save_png(ref_proj, f"{out_dir}RAW/proj_{ref}_fiducial.png")
        save_png(target_proj, f"{out_dir}RAW/proj_{cycle}_fiducial.png")
        # registration global
        shift_values, _, _ = phase_cross_correlation(
            ref_proj, target_proj, upsample_factor=100
        )
        print(f"shift values: {shift_values}")
        shifted_target = shift(target_proj, shift_values)
        # saves output data
        save_png(ref_proj, f"{out_dir}registered/proj_{ref}_ref.png")
        save_png(shifted_target, f"{out_dir}registered/proj_{cycle}_registered.png")


def main():
    folder = "/home/xdevos/grey/ProcessedData_2024/Experiment_49_David_RAMM_DNAFISH_Bantignies_proto_G1E_cells_LRKit/deinterleave_deconvolved_test/005"
    cycle_list = ["DAPI", "RT17"]
    extract_cell_scale_tif(folder, cycle_list, 924)
    global_align(cycle_list)


if __name__ == "__main__":
    begin_time = datetime.now()
    main()
    print("\n==================== Normal termination ====================\n")
    print(f"Elapsed time: {datetime.now() - begin_time}")
