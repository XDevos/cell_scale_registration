#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from data_manager import init_localization_table


def update_zxy_to_row(localization_raw, new_z, new_x, new_y):
    localization_raw["zcentroid"] = new_z
    localization_raw["xcentroid"] = new_x
    localization_raw["ycentroid"] = new_y
    return localization_raw


def coord_in_shape(shape, z, x, y):
    depth, height, width = shape
    if z < 0 or z >= depth or x < 0 or x >= height or y < 0 or y >= width:
        return False
    return True


def register_localizations(spots_3d, mask_3d, registration_table):
    new_spots_3d = init_localization_table()
    for spot in spots_3d:
        z = spot["zcentroid"]
        x = spot["xcentroid"]
        y = spot["ycentroid"]
        if not coord_in_shape(mask_3d.shape, z, x, y):
            continue
        id = mask_3d[z, x, y]
        if id != 0:
            z_global_shift = registration_table[id]["z_global_shift"]
            x_global_shift = registration_table[id]["x_global_shift"]
            y_global_shift = registration_table[id]["y_global_shift"]
            zoom_factor = registration_table[id]["zoom_factor"]
            x_centroid = registration_table[id]["x_centroid"]
            y_centroid = registration_table[id]["y_centroid"]
            x_final_shift = registration_table[id]["x_final_shift"]
            y_final_shift = registration_table[id]["y_final_shift"]
            new_z = z + z_global_shift
            new_x = (
                x_centroid
                + (x_global_shift + x - x_centroid) / zoom_factor
                + x_final_shift
            )
            new_y = (
                y_centroid
                + (y_global_shift + y - y_centroid) / zoom_factor
                + y_final_shift
            )
            row = update_zxy_to_row(spot, new_z, new_x, new_y)
            new_spots_3d.add_row(row)
    return new_spots_3d
