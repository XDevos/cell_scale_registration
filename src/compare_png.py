#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def compare_images(image1_path, image2_path):
    # Charger les deux images avec Pillow et les convertir en niveaux de gris
    image1 = Image.open(image1_path).convert("L")
    image2 = Image.open(image2_path).convert("L")

    # Convertir les images en tableaux NumPy
    image1_np = np.array(image1)
    image2_np = np.array(image2)

    # Vérifier que les deux images ont la même taille
    if image1_np.shape != image2_np.shape:
        raise ValueError("Les images doivent avoir la même taille pour être comparées.")

    # Calculer le SSIM entre les deux images
    score, diff = ssim(image1_np, image2_np, full=True)

    # Différence normalisée entre les deux images
    diff = (diff * 255).astype("uint8")

    return score, diff


# Utilisation de la fonction
# folder = "/home/xdevos/Repositories/XDevos/cell_scale_registration/src/OUT/global_registration/"
# image1_path = folder + "proj_DAPI_ref.png"
# image2_path = folder + "proj_RT17_registered.png"

folder = "/home/xdevos/Repositories/XDevos/cell_scale_registration/src/OUT/global_registration/RAW/"
image1_path = folder + "proj_DAPI_fiducial.png"
image2_path = folder + "proj_RT17_fiducial.png"

score, diff = compare_images(image1_path, image2_path)

print(f"Score de similarité (SSIM): {score}")


def show_difference(diff):
    plt.imshow(diff, cmap="gray")
    plt.title("Différence entre les images")
    plt.show()


# Afficher la différence
show_difference(diff)
