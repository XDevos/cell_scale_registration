#!/usr/bin/env python3
# -*- coding: utf-8 -*-

mask_3d = load_npy(mask_3d_file)
dapi_3d = load_tif(dapi_3d_file)
dapi_fiducial_3d = load_tif(dapi_fiducial_3d_file)
filtered_mask_3d = filter_masks_by_intensity_max(mask_3d, dapi_3d) # or filter with fiducial of dapi ?
mask_props = props = measure.regionprops(filtered_mask_3d)
rt_fiducial_file_list = get_fiducial_files()

# DAPI mask registration
for cycle_file in rt_fiducial_file_list:
    mask_3d_for_cycle = np.zeros(mask_3d.shape)
    target_fiducial_3d = load_tif(cycle_file)
    for props in mask_props:
        bbox_ref = extract_masked_tif(dapi_fiducial_3d, props.bbox)
        bbox_target_raw = extract_masked_tif(target_fiducial_3d, props.bbox)
        # X-correlation on 3 axis
        global_shift, _, _ = phase_cross_correlation(bbox_ref, bbox_target_raw, upsample_factor=100)
        shifted_bbox = shift_bbox(props.bbox, shift_values)
        bbox_target = extract_masked_tif(target_fiducial_3d, shifted_bbox)
        ref_2d = projection_mip(bbox_ref)
        target_2d = projection_mip(bbox_target)
        zoom_factor = register_translated_polar(ref_2d, target_2d)
        if zoom_factor > 1:
            continue
        scaled_target_2d = zoom_on_center(target_2d, zoom_factor)
        final_shift,_,_ =  phase_cross_correlation(ref_2d, scaled_target_2d, upsample_factor=100)
        save_registration(cycle_name, label_bnr, global_shift,zoom_factor, props.centroid, final_shift)
        plot_comparaison(ref_2d, scaled_target_2d)
        mask_3d_for_cycle = generate_mask(props, global_shift,zoom_factor, final_shift)
    save_npy(mask_3d_for_cycle, mask_3d_for_cycle_filepath)    

spots_3d = load_ecsv(localizations_file)
new_spots_3d = astropy_table()
registration_info = load_ecsv(registration_file)
spots_by_cycle = group_by_cycle(spots_3d)
for cycle in spots_by_cycle:
    mask_3d_for_cycle = load_npy(find_path(cycle))
    for spot in cycle:
        id = mask_3d_for_cycle[spot.z,spot.x,spot.y]
        if id != 0:
            global_shift,zoom_factor, centroid, final_shift = registration_info[cycle.name][id]
            new_x = centroid.x + ((global_shift.x + spot.x) - centroid.x) * zoom_factor + final_shift.x
            new_y = centroid.y + ((global_shift.y + spot.y) - centroid.y) * zoom_factor + final_shift.y
            new_z = global_shift.z + spot.z
        new_spots_3d.append(spot.info, new_x, new_y, new_z)
save_ecsv(new_spots_3d, new_spots_3d_path)