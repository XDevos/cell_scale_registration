#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage import io


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
