#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, regionprops_table, label
from tifffile import imread
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks

print("loading...")
# Charger les images DAPI et la segmentation
folder = "/home/xdevos/grey/ProcessedData_2024/Experiment_49_David_RAMM_DNAFISH_Bantignies_proto_G1E_cells_LRKit/deinterleave_deconvolved_test/005/"
dapi_image = imread(folder + "scan_001_DAPI_005_ROI_converted_decon_ch00.tif")
print("dapi tif loaded.")
segmentation_image = np.load(
    folder + "mask_3d/data/scan_001_DAPI_005_ROI_converted_decon_ch00_3Dmasks.npy"
)
print("mask npy loaded.")


# Vérifier que les deux images ont la même forme
if dapi_image.shape != segmentation_image.shape:
    raise ValueError("Les images DAPI et de segmentation doivent avoir la même forme.")

# # Créer un label pour chaque région dans la segmentation
# labeled_image = label(segmentation_image)

# Initialiser les listes pour les statistiques des masques 3D
intensity_max_values = []
region_labels = []  # Liste des labels de chaque région pour référence

# Calculer les statistiques pour chaque masque 3D
for region in tqdm(
    regionprops(segmentation_image, intensity_image=dapi_image),
    desc="Traitement des masques 3D",
):
    # Intensité maximale et label
    intensity_max_values.append(np.max(region.intensity_image))
    region_labels.append(region.label)

# Générer l'histogramme de l'intensité maximale pour détection des pics
counts, bins = np.histogram(intensity_max_values, bins=30)
peaks, _ = find_peaks(counts, height=1)

# Vérifier qu'il y a au moins deux pics pour identifier un creux
if len(peaks) < 2:
    raise ValueError(
        "Moins de deux pics trouvés, impossible de déterminer un creux entre deux pics."
    )

# Trouver le premier creux entre les deux premiers pics
first_peak, second_peak = peaks[:2]
creux_index = np.argmin(counts[first_peak:second_peak]) + first_peak
creux_intensity = bins[creux_index]

# Supprimer les masques dont l'intensité maximale est inférieure au creux
new_segmentation = np.zeros_like(segmentation_image)
for max_intensity, label_id in zip(intensity_max_values, region_labels):
    if max_intensity >= creux_intensity:
        new_segmentation[segmentation_image == label_id] = label_id

# Sauvegarder le nouveau fichier de masques 3D en npy
np.save("new_segmentation.npy", new_segmentation)

# Affichage des informations
print(
    f"Creux entre les deux premiers pics situé à une intensité maximale de : {creux_intensity}"
)
print("Nouveau fichier de masques sauvegardé sous le nom 'new_segmentation.npy'.")
