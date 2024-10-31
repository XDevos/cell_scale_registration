#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from project import mip_projection
from compare import plot_comparaison
from skimage.measure import regionprops
from skimage.registration import phase_cross_correlation
from astropy.table import Table
from skimage.filters import window, difference_of_gaussians
from skimage.transform import warp_polar, SimilarityTransform, warp
from scipy.fft import fft2, fftshift
from tqdm import tqdm
from scipy.signal import find_peaks

from data_manager import init_registration_table


def filter_masks_by_intensity_max(dapi_3d_mask, dapi_3d):
    if dapi_3d_mask.shape != dapi_3d.shape:
        raise ValueError("Shape error")

    intensity_max_values = []
    region_labels = []

    for region in tqdm(
        regionprops(dapi_3d_mask, intensity_image=dapi_3d),
        desc="Traitement des masques 3D",
    ):
        intensity_max_values.append(np.max(region.intensity_image))
        region_labels.append(region.label)
    counts, bins = np.histogram(intensity_max_values, bins=30)
    peaks, _ = find_peaks(counts, height=1)

    if len(peaks) < 2:
        raise ValueError(
            "Fewer than two peaks found, impossible to determine a trough between two peaks."
        )

    # Find first trough
    first_peak, second_peak = peaks[:2]
    creux_index = np.argmin(counts[first_peak:second_peak]) + first_peak
    creux_intensity = bins[creux_index]

    # Remove masks without enough intensity
    new_segmentation = np.zeros_like(dapi_3d_mask)
    for max_intensity, label_id in zip(intensity_max_values, region_labels):
        if max_intensity >= creux_intensity:
            new_segmentation[dapi_3d_mask == label_id] = label_id

    return new_segmentation


def get_relevant_masks_info(dapi_3d_mask, dapi_3d):
    # or filter with fiducial of dapi ?
    filtered_mask_3d = filter_masks_by_intensity_max(dapi_3d_mask, dapi_3d)
    mask_props = regionprops(filtered_mask_3d)
    return mask_props


def extract_masked_tif(data_3d, bbox):
    # TODO: may be take the real mask (like a sphere and not just bbox of mask)
    minz, minx, miny, maxz, maxx, maxy = bbox
    return data_3d[minz:maxz, minx:maxx, miny:maxy]


def get_global_shift(dapi_fiducial_3d, target_fiducial_3d, bbox):
    bbox_ref = extract_masked_tif(dapi_fiducial_3d, bbox)
    bbox_target_raw = extract_masked_tif(target_fiducial_3d, bbox)
    # X-correlation on 3 axis
    global_shift, _, _ = phase_cross_correlation(
        bbox_ref, bbox_target_raw, upsample_factor=100
    )
    return global_shift, bbox_ref


def zoom_on_center(target, shift_scale):
    shape = target.shape
    center = (shape[0] / 2, shape[1] / 2)
    translation_center = (
        -center[0] * (shift_scale - 1),
        -center[1] * (shift_scale - 1),
    )
    similarity_transform = SimilarityTransform(
        scale=shift_scale, translation=translation_center
    )
    return warp(target, similarity_transform, output_shape=shape)


def get_final_shift(ref_2d, target_2d, zoom_factor):
    scaled_target_2d = zoom_on_center(target_2d, zoom_factor)
    final_shift, _, _ = phase_cross_correlation(
        ref_2d, scaled_target_2d, upsample_factor=100
    )
    return final_shift


def shift_bbox(bbox, shift):
    dim = len(shift)
    if len(bbox) != dim * 2:
        raise ValueError(
            f"Uncompatible dimension for shift ({shift}) and bbox ({bbox})."
        )
    new_bbox = np.zeros(len(bbox))
    for axis in range(dim):
        new_bbox[axis] = bbox[axis] + shift[axis]
        new_bbox[axis + dim] = bbox[axis + dim] + shift[axis]
    return tuple(new_bbox)


def get_shifted_target_2d(target_fiducial_3d, bbox, global_shift):
    shifted_bbox = shift_bbox(bbox, global_shift)
    bbox_target = extract_masked_tif(target_fiducial_3d, shifted_bbox)
    return mip_projection(bbox_target)


def register_translated_polar(ref_2d, target_2d):
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
    shift_scale = np.exp(shiftc / klog)
    return shift_scale


def add_mask(mask_3d, props, global_shift, zoom_factor, final_shift):
    centroid = props.centroid
    for z, x, y in props.coords:
        new_z = z - global_shift[0]
        new_x = (
            (x - centroid[1] - final_shift[0]) / zoom_factor
            + centroid[1]
            - global_shift[1]
        )
        new_y = (
            (y - centroid[2] - final_shift[1]) / zoom_factor
            + centroid[2]
            - global_shift[2]
        )
        mask_3d[int(new_z)][int(new_x)][int(new_y)] = props.label
    return mask_3d


def register_by_dapi_mask(mask_props, dapi_fiducial_3d, target_fiducial_3d, cycle_name):
    mask_3d_for_cycle = np.zeros(target_fiducial_3d.shape)
    registration_table = init_registration_table()
    for props in mask_props:
        global_shift, bbox_ref = get_global_shift(
            dapi_fiducial_3d, target_fiducial_3d, props.bbox
        )
        ref_2d = mip_projection(bbox_ref)
        target_2d = get_shifted_target_2d(target_fiducial_3d, props.bbox, global_shift)
        zoom_factor = register_translated_polar(ref_2d, target_2d)
        final_shift = get_final_shift(ref_2d, target_2d, zoom_factor)
        if zoom_factor < 1:
            registration_table.add_row(
                [
                    cycle_name,
                    props.label,
                    global_shift[0],
                    global_shift[1],
                    global_shift[2],
                    zoom_factor,
                    props.centroid[0],
                    props.centroid[1],
                    props.centroid[2],
                    final_shift[0],
                    final_shift[1],
                ]
            )
            mask_3d_for_cycle = add_mask(
                mask_3d_for_cycle, props, global_shift, zoom_factor, final_shift
            )
    return mask_3d_for_cycle, registration_table
