import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


def print_ssim_mse(ref, targ):
    ssim_none = ssim(ref, targ)
    print(f"SSIM: {ssim_none}")
    mse_none = mean_squared_error(ref, targ)
    print(f"MSE: {mse_none}")

def preprocess_image(image_array):
    # Assurez-vous que l'image est en 2D (grayscale)
    if image_array.ndim == 3:
        # Convertir en niveaux de gris en prenant la moyenne des canaux
        print("WARNING project in 2D with mean")
        gray_image = np.mean(image_array, axis=2)
    else:
        gray_image = image_array

    # Histogram Equalization
    equalized_image = exposure.equalize_hist(gray_image)

    # Réduire le bruit avec un filtre médian
    denoised_image = filters.median(gray_image)
    denoised_image = gray_image

    # Normaliser l'image
    print(f"Normalized factor:\ndenoised_image - {np.min(denoised_image)}) / ({np.max(denoised_image)} - {np.min(denoised_image)}")
    normalized_image = (denoised_image - np.min(denoised_image)) / (np.max(denoised_image) - np.min(denoised_image))
    print(f"Normalized factor:\ndenoised_image - {np.min(normalized_image)}) / ({np.max(normalized_image)} - {np.min(normalized_image)}")
    return normalized_image

# Chemins des fichiers NPY
fold = "/home/xdevos/Repositories/XDevos/explore_registration/sample_924/8_global_shift/mip_projection/"
image1_path = fold +  'ref.npy'
image2_path = fold + 'target.npy'

# Charger les images à partir des fichiers NPY
image1 = np.load(image1_path)
image2 = np.load(image2_path)

# Prétraiter les images
preprocessed_image1 = preprocess_image(image1)
preprocessed_image2 = preprocess_image(image2)
print_ssim_mse(preprocessed_image1, preprocessed_image2)

def show_both_img(im1, im2):
    # Afficher les images prétraitées
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Image 1 Prétraitée')
    plt.imshow(im1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Image 2 Prétraitée')
    plt.imshow(im2, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

show_both_img(image1,image2)
show_both_img(preprocessed_image1,preprocessed_image2)