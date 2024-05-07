
## Dependencies
- os
- cv2
- numpy
- tensorflow
- skimage
- scipy
- sklearn
- matplotlib
- lpips
- torch
- transformers
- pandas

## Code


import os
import cv2
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
from sklearn.random_projection import GaussianRandomProjection
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import lpips
import torch
from sklearn.metrics.pairwise import rbf_kernel
from skimage.util import view_as_windows
from sklearn.metrics.pairwise import rbf_kernel
from transformers import ViTFeatureExtractor, ViTModel
from tensorflow.keras.applications import ResNet50
from scipy.stats import entropy
from scipy.linalg import sqrtm
import pandas as pd
folders = ['Original', 'Stable_Diffusion', 'DALL-E 2', 'GLIDE', 'DALL-E 3']

inception_model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet', input_shape=(299, 299, 3))
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

desired_size = (299, 299)

def read_and_resize(folder, ext):
    images = {}
    for file in os.listdir(folder):
        if file.endswith(ext):
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.resize(img, desired_size)
            images[file.split('.')[0]] = img
    return images

originals = read_and_resize(folders[0], '.jpg')
stable_diffs = read_and_resize(folders[1], '.png')
dells2 = read_and_resize(folders[2], '.png')
glides = read_and_resize(folders[3], '.png')
dells3 = read_and_resize(folders[4], '.png')

def display_all_images(images_dict, title):
    print(f"Total images in {title}: {len(images_dict)}")
    for image_key in images_dict:
        plt.figure()
        plt.imshow(cv2.cvtColor(images_dict[image_key], cv2.COLOR_BGR2RGB))
        plt.title(f"{title} - {image_key}")
        plt.axis('off')
        plt.show()

display_all_images(originals, "Originals")
display_all_images(stable_diffs, "Stable Diffusion")
display_all_images(dells2, "DALL-E 2")
display_all_images(glides, "GLIDE")
display_all_images(dells3, "DALL-E 3")

def extract_patches(image, patch_size):
    """
    Extracts square patches from an image.
    """
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    patches = view_as_windows(image, window_shape=(patch_size, patch_size, image.shape[2]), step=patch_size)
    return patches.reshape(-1, patch_size, patch_size, image.shape[2])


def extract_relevant_patches(image, patch_size, top_k=5):
    """
    Extracts the most relevant patches based on transformer attention.
    """
    inputs = feature_extractor(images=image, return_tensors="pt") 
    outputs = model(**inputs)
    attention_weights = outputs.last_hidden_state.mean(axis=1)  
    attention_weights = attention_weights.detach().numpy()  

    # Extract patches
    patches = view_as_windows(image, window_shape=(patch_size, patch_size, image.shape[2]), step=patch_size)
    patches_flat = patches.reshape(-1, patch_size, patch_size, image.shape[2])

    # Get indices of top-k attention weights and ensure they are within bounds
    top_patches_indices = np.argsort(attention_weights)[-top_k:]
    top_patches_indices = top_patches_indices[top_patches_indices < len(patches_flat)]  

    return patches_flat[top_patches_indices]

def dice_coefficient(patch1, patch2):
    """
    Computes the Dice coefficient for two patches.
    Adapts the Dice calculation for continuous data.
    """
    patch1_flat = patch1.flatten()
    patch2_flat = patch2.flatten()
    
    intersection = np.sum(patch1_flat * patch2_flat)
    return 2. * intersection / (np.sum(patch1_flat) + np.sum(patch2_flat))

def calculate_patch_dice_similarity(images_real, images_fake, patch_size):
    """
    Calculates the average similarity between pairs of real and fake images based on patches.
    """
    dice_scores = []
    for img_real, img_fake in zip(images_real, images_fake):
        patches_real = extract_relevant_patches(img_real, patch_size)
        patches_fake = extract_relevant_patches(img_fake, patch_size)
        # Assuming an equal number of patches from real and fake images
        scores = [dice_coefficient(p_real, p_fake) for p_real, p_fake in zip(patches_real, patches_fake)]
        dice_scores.append(np.mean(scores))
    return np.mean(dice_scores)  

def DVPI(images_real, images_fake, model, base_gamma, patch_size=16):
    """
    Computes DVPI while considering patch-based similarity between images.
    """
    images_real_array = np.array(list(images_real.values()))
    images_fake_array = np.array(list(images_fake.values()))
    
    act_real = model.predict(images_real_array)
    act_fake = model.predict(images_fake_array)

    # Dynamically adjust gamma based on content complexity
    complexity_real = np.var(act_real)
    complexity_fake = np.var(act_fake)
    avg_complexity = (complexity_real + complexity_fake) / 2
    gamma = base_gamma * np.sqrt(avg_complexity) + 0.20

    # Calculate pairwise kernels with adjusted gamma
    K_xx = rbf_kernel(act_real, act_real, gamma)
    K_yy = rbf_kernel(act_fake, act_fake, gamma)
    K_xy = rbf_kernel(act_real, act_fake, gamma)

    # Calculate the Maximum Mean Discrepancy (MMD)
    N = act_real.shape[0]
    M = act_fake.shape[0]
    mmd = (np.sum(K_xx) / (N**2)) + (np.sum(K_yy) / (M**2)) - (2 * np.sum(K_xy) / (N * M))
    print(mmd)
    # Calculate the average Dice similarity for patches
    avg_dice_similarity = calculate_patch_dice_similarity(images_real_array, images_fake_array, patch_size)
    print(avg_dice_similarity)
    lambda_param = 0.50 
    score = avg_dice_similarity * (1 - lambda_param * mmd)  
    return score

# Compute  scores
kid_stable_diffs = DVPI(originals, stable_diffs, inception_model, 4)
kid_dells = DVPI(originals, dells2, inception_model,4)
kid_glides = DVPI(originals, glides, inception_model,4)
kid_dells2 = DVPI(originals, dells3, inception_model,4)

# Print DVPI scores
print(f"DVPI RBF between Originals and Stable Diffusion: {kid_stable_diffs}")
print(f"DVPI RBF between Originals and DALL-E 2: {kid_dells}")
print(f"DVPI RBF between Originals and GLIDE: {kid_glides}")
print(f"DVPI RBF between Originals and DALL-E 3: {kid_dells2}")
