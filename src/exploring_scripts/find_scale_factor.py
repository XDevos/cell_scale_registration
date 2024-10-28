#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from skimage.transform import rescale, SimilarityTransform, warp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

folder = "/home/xdevos/Repositories/XDevos/cell_scale_registration/src/OUT/global_registration/registered/"

path1 = folder + "proj_DAPI_ref.png"
path2 = folder + "proj_RT17_registered.png"

image = np.array(Image.open(path1).convert("L"))
rescaled = np.array(Image.open(path2).convert("L"))
height, _ = image.shape
radius = (height // 2) * 0.8

# radius must be large enough to capture useful info in larger image
image_polar = warp_polar(image, radius=radius, scaling="linear")
rescaled_polar = warp_polar(rescaled, radius=radius, scaling="linear")

# fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# ax = axes.ravel()
# ax[0].set_title("Original")
# ax[0].imshow(image)
# ax[1].set_title("Rotated and Rescaled")
# ax[1].imshow(rescaled)
# ax[2].set_title("Log-Polar-Transformed Original")
# ax[2].imshow(image_polar)
# ax[3].set_title("Log-Polar-Transformed Rotated and Rescaled")
# ax[3].imshow(rescaled_polar)
# plt.show()

# setting `upsample_factor` can increase precision
shifts, error, phasediff = phase_cross_correlation(
    image_polar, rescaled_polar, upsample_factor=20
)
shiftr, shiftc = shifts[:2]

# Calculate scale factor from translation
klog = radius / np.log(radius)
shift_scale = np.exp(shiftc / klog)

print(f"Recovered value for cc rotation: {shiftr}")
print()
print(f"Recovered value for scaling difference: {shift_scale}")

# Rescale the image using scikit-image
tform = SimilarityTransform(scale=shift_scale)
print(tform.params)
tf_img = warp(rescaled, tform.inverse)


def save_png(data, output_path):
    fig = plt.figure()
    size = (10, 10)
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_title("2D Data")
    fig.add_axes(ax)
    ax.imshow(data, origin="lower", cmap="Greys_r")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"> $OUTPUT{output_path}")


save_png(tf_img, folder + "test.png")
