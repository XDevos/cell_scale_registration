#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from tifffile import imread

# Charger les images DAPI et la segmentation
folder = "/home/xdevos/grey/ProcessedData_2024/Experiment_49_David_RAMM_DNAFISH_Bantignies_proto_G1E_cells_LRKit/deinterleave_deconvolved_test/005/"
dapi_image = imread(folder + "scan_001_DAPI_005_ROI_converted_decon_ch00.tif")
# segmentation_image = np.load(
#     folder + "mask_3d/data/scan_001_DAPI_005_ROI_converted_decon_ch00_3Dmasks.npy"
# )
segmentation_image = np.load("new_segmentation.npy")

# Vérifier que les deux images ont la même forme
if dapi_image.shape != segmentation_image.shape:
    raise ValueError("Les images DAPI et de segmentation doivent avoir la même forme.")

# # Créer un label pour chaque région dans la segmentation
# labeled_image = label(segmentation_image)

# Calculer les propriétés de chaque région
regions = regionprops(segmentation_image, intensity_image=dapi_image)

# Initialiser des listes pour stocker les valeurs de chaque masque
intensity_profiles = []
centroid_positions = []
pixel_counts = []

for region in regions[:10]:
    print("é")
    # Calcul de l'intensité moyenne par plan (profil d'intensité)
    intensity_profile = [
        np.mean(dapi_image[plane][region.coords[:, 0], region.coords[:, 1]])
        for plane in range(dapi_image.shape[0])
    ]
    intensity_profiles.append(intensity_profile)

    # Position du centroïde en Z
    centroid_positions.append(region.centroid[0])  # Coordonnée Z du centroïde

    # Nombre de pixels par masque
    pixel_counts.append(region.area)

# Générer les courbes
plt.figure(figsize=(15, 5))

# Courbe d'intensité
plt.subplot(1, 3, 1)
for profile in intensity_profiles:
    plt.plot(profile)
plt.title("Courbe d'intensité de chaque masque")
plt.xlabel("Plan Z")
plt.ylabel("Intensité moyenne")

# Courbe des positions des centroïdes sur l'axe Z
plt.subplot(1, 3, 2)
plt.plot(centroid_positions, "o-")
plt.title("Positions des centroïdes sur l'axe Z")
plt.xlabel("Masque")
plt.ylabel("Position Z du centroïde")

# Courbe de répartition du nombre de pixels par masque
plt.subplot(1, 3, 3)
plt.plot(pixel_counts, "o-")
plt.title("Répartition du nombre de pixels par masque")
plt.xlabel("Masque")
plt.ylabel("Nombre de pixels")

plt.tight_layout()
plt.show()
