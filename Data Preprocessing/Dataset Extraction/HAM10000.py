import kagglehub
import shutil
import os

# Download the dataset
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

# Define the target directory
target_dir = r'C:\Users\shore\Desktop\APS360\Datasets\HAM10000'
os.makedirs(target_dir, exist_ok=True)

# Move the entire downloaded dataset to your desired directory
shutil.move(path, target_dir)

print("Dataset moved to:", target_dir)




