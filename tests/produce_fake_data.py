#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tifffile as tiff

dapi_mask_3d = np.zeros((60, 100, 100), dtype=np.uint8)
dapi_mask_3d[20:40, 10:90, 10:90] = 1
np.save("dapi_mask_3d2.npy", dapi_mask_3d)


dapi_3d = np.ones((60, 100, 100), dtype=np.uint8) * 32
dapi_3d[20:40, 10:90, 10:90] = 255
tiff.imwrite("dapi_3d2.tif", dapi_3d)

fidu_dapi = np.ones((60, 100, 100), dtype=np.uint8) * 32
# fidu_dapi[20:40, 30:70, 30:34] = 255
# fidu_dapi[20:40, 30:70, 66:70] = 255
# fidu_dapi[20:40, 30:34, 30:70] = 255
# fidu_dapi[20:40, 66:70, 30:70] = 255
fidu_dapi[20:40, 48:52, 30:70] = 255
fidu_dapi[20:40, 30:70, 48:52] = 255
tiff.imwrite("fidu_dapi2.tif", fidu_dapi)

fidu_rt = np.ones((60, 100, 100), dtype=np.uint8) * 32
# fidu_rt[20:40, 30:50, 30:32] = 255
# fidu_rt[20:40, 30:50, 48:50] = 255
# fidu_rt[20:40, 30:32, 30:50] = 255
# fidu_rt[20:40, 48:50, 30:50] = 255
fidu_rt[10:30, 39:41, 30:50] = 255
fidu_rt[10:30, 30:50, 39:41] = 255
tiff.imwrite("fidu_rt2.tif", fidu_rt)
