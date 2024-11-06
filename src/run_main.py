#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

from run_args import parse_run_args
from data_manager import DataManager
from registration_by_mask import get_relevant_masks_info, register_by_dapi_mask
from spot_registration import register_localizations


def main():
    run_args = parse_run_args()
    datam = DataManager(run_args.input, run_args.output)

    # DAPI mask registration
    # mask_props = get_relevant_masks_info(datam.dapi_3d_mask, datam.dapi_3d)
    # for fiducial_file in datam.target_fiducial_files:
    #     target_fiducial, cycle_name = datam.load_target_fiducial(fiducial_file)
    #     target_mask_3d, registration_table = register_by_dapi_mask(
    #         mask_props, datam.dapi_fiducial_3d, target_fiducial, cycle_name
    #     )
    #     datam.save_mask_3d_for_cycle(target_mask_3d, cycle_name)
    #     datam.update_registration_info(registration_table)

    # Register localizations
    for cycle in datam.cycle_list:
        # continue  # tempo
        mask_3d = datam.get_mask_3d_for_cycle(cycle)
        raw_spots_3d = datam.get_raw_spots_3d_by_cycle(cycle)
        registration_info = datam.get_registration_info_by_cycle(cycle)
        new_spots_3d = register_localizations(raw_spots_3d, mask_3d, registration_info)
        datam.update_registered_localizations(new_spots_3d, cycle)


if __name__ == "__main__":
    begin_time = datetime.now()
    main()
    print("\n==================== Normal termination ====================\n")
    print(f"Elapsed time: {datetime.now() - begin_time}")
