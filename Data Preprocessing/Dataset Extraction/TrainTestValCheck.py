import os
import torch
import numpy as np

# --- Paths ---
split_dir = r'C:\Users\shore\Desktop\APS360\Datasets\DataSplit'

# --- Label Mapping ---
label_mapping = {
    'nevus': 0,
    'melanoma': 1,
    'bcc': 2,
    'seborrheic_keratosis': 3,
    'actinic_keratosis': 4,
    'scc': 5,
    'dermatofibroma': 6,
    'lentigo': 7,
    'vascular_lesion': 8
}

# --- Inverse Mapping for Readability ---
inv_label_mapping = {v: k for k, v in label_mapping.items()}

def check_distribution(split_name):
    split_path = os.path.join(split_dir, split_name)
    class_counts = {label: 0 for label in label_mapping.keys()}

    for label in os.listdir(split_path):
        label_dir = os.path.join(split_path, label)
        if os.path.isdir(label_dir):
            num_images = len(os.listdir(label_dir))
            class_counts[label] += num_images

    print(f"\nClass distribution in {split_name} split:")
    total = sum(class_counts.values())
    for label, count in class_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{label} -> {count} images ({percentage:.2f}%)")
    print(f"Total images: {total}")

# --- Run for Each Split ---
for split in ['train', 'val', 'test']:
    check_distribution(split)
