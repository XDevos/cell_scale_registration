#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tifffile as tiff

# Create an array with shape (6, 10, 10) initialized to zeros
image = np.zeros((6, 10, 10), dtype=np.uint8)

# Set the values in the specified region [2:4, 3:7, 3:7] to 1
image[2:4, 1:9, 1:9] = 1

# Save the image as a .tif file
# tiff.imwrite("generated_image.tif", image)
np.save("dapi_mask_3d.npy", image)

print("Image generated and saved as 'generated_image.tif'")
