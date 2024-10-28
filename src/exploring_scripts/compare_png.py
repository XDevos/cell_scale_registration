#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def extract_circle(image_path):
    # Charger l'image en niveaux de gris
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)

    # Obtenir les dimensions de l'image
    height, width = image_np.shape
    center = [height / 2, width / 2]
    radius = min(height // 2, width // 2) * 0.8
    # Créer un masque circulaire
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius

    # Appliquer le masque circulaire
    circular_image = np.zeros_like(image_np)
    circular_image[mask] = image_np[mask]

    return circular_image


def compare_images(image1_path, image2_path):
    # # Charger les deux images avec Pillow et les convertir en niveaux de gris
    # image1 = Image.open(image1_path).convert("L")
    # image2 = Image.open(image2_path).convert("L")

    # # Convertir les images en tableaux NumPy
    # image1_np = np.array(image1)
    # image2_np = np.array(image2)

    image1_np = extract_circle(image1_path)
    image2_np = extract_circle(image2_path)

    # Vérifier que les deux images ont la même taille
    if image1_np.shape != image2_np.shape:
        raise ValueError("Les images doivent avoir la même taille pour être comparées.")

    # Calculer le SSIM entre les deux images
    score, diff = ssim(image1_np, image2_np, full=True)

    # Différence normalisée entre les deux images
    diff = (diff * 255).astype("uint8")

    return score, diff


# Utilisation de la fonction
folder = "/home/xdevos/Repositories/XDevos/cell_scale_registration/src/OUT/global_registration/registered/"
image1_path = folder + "proj_DAPI_ref.png"
image2_path = folder + "proj_RT17_registered.png"

score, diff = compare_images(image1_path, image2_path)

print(f"Score de similarité (SSIM): {score}")


def show_difference(diff):
    plt.imshow(diff, cmap="gray")
    plt.title("Différence entre les images")
    plt.show()


# Afficher la différence
show_difference(diff)

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
