#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage import io
from astropy.table import Table, vstack


def load_tif(path):
    data = io.imread(path)
    print(f"TIF loaded: {path}")
    return data


def save_tif(path, data):
    io.imsave(path, data)
    print(f"TIF saved: {path}")


def load_npy(path):
    data = np.load(path)
    print(f"NPY loaded: {path}")
    return data


def save_npy(path, data):
    np.save(path, data)
    print(f"NPY saved: {path}")


def save_png(path, data):
    if path[-3:] != "png":
        s_path = path.split(".")
        print(f"Remove extension '{s_path.pop()}' by 'png'")
        s_path.append("png")
        path = ".".join(s_path)
    io.imsave(path, data)
    print(f"PNG saved: {path}")


def load_ecsv(file_path):
    table = Table.read(file_path, format="ascii.ecsv")
    print(f"[Load] {file_path}")
    return table


def save_ecsv(table, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    table.write(path, format="ascii.ecsv", overwrite=True)
    print(f"[Saving] {path}")


def save_to_compare(masked_ref, masked_target, sub_folder):
    folder = "/home/xdevos/Repositories/XDevos/explore_registration/sample_924/"
    os.makedirs(os.path.join(folder, sub_folder), exist_ok=True)
    if len(masked_ref.shape) == 2:
        ref_path = os.path.join(folder, sub_folder, "ref.npy")
        save_npy(ref_path, masked_ref)
        target_path = os.path.join(folder, sub_folder, "target.npy")
        save_npy(target_path, masked_target)
        save_png(ref_path, masked_ref)
        save_png(target_path, masked_target)
    elif len(masked_ref.shape) == 3:
        ref_path = os.path.join(folder, sub_folder, "ref.tif")
        save_tif(ref_path, masked_ref)
        target_path = os.path.join(folder, sub_folder, "target.tif")
        save_tif(target_path, masked_target)
    else:
        raise ValueError(f"Bad data dimension; masked_ref.shape == {masked_ref.shape}")


def extract_files(root: str):
    """Extract recursively file informations of all files into a given directory.
    Note:
    * filepath is directory path with filename and extension
    * filename is the name without extension

    Parameters
    ----------
    root : str
        The name of root directory

    Returns
    -------
    List[Tuple(str,str,str)]
        List of file informations: (filepath, filename, extension)
    """
    files = []
    # Iterate into dirname and each subdirectories dirpath, dirnames, filenames
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            split_filename = filename.split(".")
            extension = split_filename.pop() if len(split_filename) > 1 else None
            short_filename = ".".join(split_filename)
            filepath = os.path.join(dirpath, filename)
            files.append((filepath, short_filename, extension))
    return files


class DataManager:
    def __init__(self, input_folder, output_folder):
        self.in_folder = input_folder
        self.out_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.in_files = extract_files(self.in_folder)
        self.roi = "002"
        dapi_3d_mask_file = os.path.join(
            self.in_folder,
            f"scan_001_DAPI_{self.roi}_ROI_converted_decon_ch00_3Dmasks.npy",
        )
        self.dapi_3d_mask = load_npy(dapi_3d_mask_file)
        dapi_3d_file = os.path.join(
            self.in_folder, f"scan_001_DAPI_{self.roi}_ROI_converted_decon_ch00.tif"
        )
        self.dapi_3d = load_tif(dapi_3d_file)
        dapi_fiducial_3d_file = os.path.join(
            self.in_folder, f"scan_001_DAPI_{self.roi}_ROI_converted_decon_ch01.tif"
        )
        self.dapi_fiducial_3d = load_tif(dapi_fiducial_3d_file)

        self.registration_info = init_registration_table()
        self.registered_spots = init_localization_table()
        self.cycle_list = [
            f"RT{i}"
            for i in [
                15,
                16,
                17,
                19,
                20,
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
        self.target_fiducial_files = self.get_fiducial_files()
        print(f"Fiducial files = \n{self.target_fiducial_files}")

    def get_fiducial_files(self):
        return [
            os.path.join(
                self.in_folder,
                f"scan_001_{cycle}_{self.roi}_ROI_converted_decon_ch00.tif",
            )
            for cycle in self.cycle_list
        ]

    def load_target_fiducial(self, file):
        target = load_tif(file)
        cycle = file.split("/")[-1].split("_")[2]
        return target, cycle

    def save_mask_3d_for_cycle(self, mask_3d, cycle):
        filepath = os.path.join(self.out_folder, f"mask_3D_{cycle}.npy")
        save_npy(filepath, mask_3d)

    def get_mask_3d_for_cycle(self, cycle):
        filepath = os.path.join(self.out_folder, f"mask_3D_{cycle}.npy")
        return load_npy(filepath)

    def update_registration_info(self, registration_table):
        self.registration_info = vstack([self.registration_info, registration_table])
        table_path = os.path.join(self.out_folder, f"registration_info.ecsv")
        save_ecsv(self.registration_info, table_path)

    def get_raw_spots_3d_by_cycle(self, cycle):
        localizations_file = os.path.join(
            self.in_folder, "localizations_3D_barcode.dat"
        )
        raw_spots_3d = load_ecsv(localizations_file)
        if cycle[:2] != "RT":
            raise ValueError(f"Cycle name ({cycle}) must start with RTxx")
        barcode_n = int(cycle[2:])
        mask = raw_spots_3d["Barcode #"] == barcode_n
        return raw_spots_3d[mask]

    def get_registration_info_by_cycle(self, cycle):
        table_path = os.path.join(self.out_folder, f"registration_info.ecsv")
        reg_info = load_ecsv(table_path)
        mask = reg_info["cycle"] == cycle
        return reg_info[mask]

    def update_registered_localizations(self, new_spots_3d):
        self.registered_spots = vstack([self.registered_spots, new_spots_3d])
        filepath = os.path.join(self.out_folder, f"registered_spots.ecsv")
        save_ecsv(self.registered_spots, filepath)


def init_registration_table():
    return Table(
        names=(
            "cycle",
            "label",
            "z_global_shift",
            "x_global_shift",
            "y_global_shift",
            "zoom_factor",
            "z_centroid",
            "x_centroid",
            "y_centroid",
            "x_final_shift",
            "y_final_shift",
        ),
        dtype=("S2", "int", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f4"),
    )


def init_localization_table():
    output = Table(
        names=(
            "Buid",
            "ROI #",
            "CellID #",
            "Barcode #",
            "id",
            "zcentroid",
            "xcentroid",
            "ycentroid",
            "sharpness",
            "roundness1",
            "roundness2",
            "npix",
            "sky",
            "peak",
            "flux",
            "mag",
        ),
        dtype=(
            "S2",
            "int",
            "int",
            "int",
            "int",
            "f4",
            "f4",
            "f4",
            "f4",
            "f4",
            "f4",
            "int",
            "f4",
            "f4",
            "f4",
            "f4",
        ),
    )
    return output
