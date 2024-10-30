#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

# Créer une image de croix simple
def create_cross_image(size):
    image = np.ones((size, size, 3))  # Image blanche
    center = size // 2
    thickness = 2  # Épaisseur de la croix

    # Dessiner la croix
    image[center - thickness:center + thickness, 3:] = [0.5, 0.5, 0.5]  # Barre horizontale
    image[:-3, center - thickness:center + thickness] = [0.5, 0.5, 0.5]  # Barre verticale
    # box
    image[-3, 3:-3] = [0, 0, 0]
    image[3, 3:-3] = [0, 0, 0]
    image[3:-3, -3] = [0, 0, 0]
    image[3:-3, 3] = [0, 0, 0]
    return image

# Dimensions de l'image
image_size = 100
image = create_cross_image(image_size)

# Afficher l'image originale
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Image Originale')
plt.imshow(image)
plt.axis('off')

# Appliquer la transformation de similarité

scale = 1.2

center = (image_size / 2, image_size / 2)  # Centre de l'image

# Créer la transformation de zoom et de translation
similarity_transform = transform.SimilarityTransform(scale=scale, translation=(-center[0] * (scale - 1), -center[1] * (scale - 1)))

# Appliquer la transformation
transformed_image = transform.warp(image, similarity_transform, output_shape=image.shape)


# Afficher l'image transformée
plt.subplot(1, 2, 2)
plt.title('Image Transformée (Scale=1.2)')
plt.imshow(transformed_image)
plt.axis('off')

# Afficher les images
plt.tight_layout()
plt.show()
