import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("mahdavi1202/skin-cancer")

# Your desired directory
target_dir = r"C:\Users\shore\Desktop\APS360\Datasets\PAD-UFES-20"

# Ensure the target directory does not already exist
if os.path.exists(target_dir):
    print(f"Target directory {target_dir} already exists. Please remove or rename it.")
else:
    # Move the dataset
    shutil.move(path, target_dir)
    print(f"Dataset moved to {target_dir}")
