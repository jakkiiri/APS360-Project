import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the labels CSV
dataset_path = r'C:\Users\shore\Desktop\APS360\Datasets\Combined'
csv_path = os.path.join(dataset_path, 'combined_labels.csv')
df = pd.read_csv(csv_path)
img_path = "C:\\Users\\shore\\Desktop\\APS360\\APS360-Project\\Data Preprocessing\\Hair Removal Images\\ik81gqr5n9l31.jpg"

# Select one image per category
sample_images = df.groupby('label').first().reset_index()

# Hair removal function
def apply_hair_removal(image, threshold, kernel_size=17):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    return inpainted, mask

# Visualization function
def visualize_hair_removal_per_class():
    thresholds = [5, 10, 20]
    for index, row in sample_images.iterrows():
        image_path = os.path.join(dataset_path, row['filename'])
        label = row['label']

        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(20, 5))
        plt.suptitle(f'Class: {label}', fontsize=16)

        # Original Image
        plt.subplot(1, len(thresholds) + 1, 1)
        plt.imshow(image_rgb)
        plt.title('Original')
        plt.axis('off')

        # Hair Removal with Different Thresholds
        for i, threshold in enumerate(thresholds):
            inpainted, _ = apply_hair_removal(image, threshold)
            plt.subplot(1, len(thresholds) + 1, i + 2)
            plt.imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
            plt.title(f'Threshold: {threshold}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

# Run visualization
#visualize_hair_removal_per_class()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_hair_removal_single_image(image_path, thresholds=[5, 10, 20], kernel_size=17):
    """
    Visualize the effect of hair removal on a single image using different thresholds.
    
    Args:
        image_path (str): Full path to the image file.
        thresholds (list): List of threshold values to test.
        kernel_size (int): Size of the morphological kernel.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare visualization
    plt.figure(figsize=(5 * (len(thresholds) + 1), 5))
    plt.suptitle(f'Hair Removal Effects: {image_path}', fontsize=16)
    
    # Show original image
    plt.subplot(1, len(thresholds) + 1, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Apply hair removal at each threshold
    for i, threshold in enumerate(thresholds):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
        inpainted = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)

        plt.subplot(1, len(thresholds) + 1, i + 2)
        plt.imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
        plt.title(f'Threshold: {threshold}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_hair_removal_single_image(img_path, thresholds=[5, 10, 20], kernel_size=17)