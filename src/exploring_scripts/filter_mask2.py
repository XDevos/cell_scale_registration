#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, regionprops_table, label
from tifffile import imread
import pandas as pd
from tqdm import tqdm

print("loading...")
# Charger les images DAPI et la segmentation
folder = "/home/xdevos/grey/ProcessedData_2024/Experiment_49_David_RAMM_DNAFISH_Bantignies_proto_G1E_cells_LRKit/deinterleave_deconvolved_test/005/"
dapi_image = imread(folder + "scan_001_DAPI_005_ROI_converted_decon_ch00.tif")
print("dapi tif loaded.")
# segmentation_image = np.load(
#     folder + "mask_3d/data/scan_001_DAPI_005_ROI_converted_decon_ch00_3Dmasks.npy"
# )
segmentation_image = np.load("new_segmentation.npy")
print("mask npy loaded.")


# Vérifier que les deux images ont la même forme
if dapi_image.shape != segmentation_image.shape:
    raise ValueError("Les images DAPI et de segmentation doivent avoir la même forme.")

# # Créer un label pour chaque région dans la segmentation
# labeled_image = label(segmentation_image)

# Initialiser les listes pour les statistiques des masques 3D
intensity_max_values = []
max_radius_values = []
centroid_z_positions = []

# Calculer les statistiques pour chaque masque 3D
for region in tqdm(
    regionprops(segmentation_image, intensity_image=dapi_image),
    desc="Traitement des masques 3D",
):
    # Intensité maximale
    intensity_max_values.append(np.max(region.intensity_image))

    # Calculer le rayon maximum en X ou Y en utilisant bbox
    min_row, min_col, min_slice, max_row, max_col, max_slice = region.bbox
    max_radius_x = (max_row - min_row) / 2
    max_radius_y = (max_col - min_col) / 2
    max_radius_values.append(max(max_radius_x, max_radius_y))

    # Position en Z du centroïde
    centroid_z_positions.append(region.centroid[0])  # La coordonnée Z du centroïde

# Générer les histogrammes
plt.figure(figsize=(20, 5))

# Histogramme de la répartition des valeurs d'intensité maximale
plt.subplot(1, 3, 1)
plt.hist(intensity_max_values, bins=30, color="green", alpha=0.7)
plt.title("Répartition des intensités maximales")
plt.xlabel("Intensité maximale")
plt.ylabel("Nombre de masques 3D")

# Histogramme de la répartition des rayons maximaux (en X ou Y)
plt.subplot(1, 3, 2)
plt.hist(max_radius_values, bins=30, color="purple", alpha=0.7)
plt.title("Répartition des rayons maximaux (X ou Y)")
plt.xlabel("Rayon maximal")
plt.ylabel("Nombre de masques 3D")

# Histogramme de la répartition des positions en Z des centroïdes
plt.subplot(1, 3, 3)
plt.hist(centroid_z_positions, bins=30, color="orange", alpha=0.7)
plt.title("Répartition des positions en Z des centroïdes")
plt.xlabel("Position Z du centroïde")
plt.ylabel("Nombre de masques 3D")

plt.tight_layout()
plt.show()
