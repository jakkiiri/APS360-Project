import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Add progress bar

# Splits data from combined.zip into train, val, and test sets with stratification and 70/15/15 ratio.

# --- Configuration ---
csv_path = r'C:\Users\shore\Desktop\APS360\Datasets\Combined\combined_labels.csv'  # Your CSV
image_root = r'C:\Users\shore\Desktop\APS360\Datasets\Combined'  # Folder with ALL images
output_dir = r'C:\Users\shore\Desktop\APS360\Datasets\DataSplit'  # Where split folders will be created

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
random_seed = 42

# --- Load Dataset ---
df = pd.read_csv(csv_path)

# --- Sanity Check ---
print(f"Total samples: {len(df)}")
print("Class distribution:\n", df['label'].value_counts())

# --- Split into Train, Val, Test (Stratified) ---
train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), stratify=df['label'], random_state=random_seed)
val_relative_ratio = val_ratio / (val_ratio + test_ratio)
val_df, test_df = train_test_split(temp_df, test_size=(1 - val_relative_ratio), stratify=temp_df['label'], random_state=random_seed)

# --- Helper Function: Copy Images to Class Folders with Progress Bar ---
def copy_images(split_df, split_name):
    for _, row in tqdm(split_df.iterrows(), total=split_df.shape[0], desc=f"Copying {split_name} images"):
        label = row['label']
        src_path = os.path.join(image_root, row['filename'])
        split_class_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(split_class_dir, exist_ok=True)
        dst_path = os.path.join(split_class_dir, os.path.basename(row['filename']))

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Missing image {src_path}")

# --- Create Folders and Copy Files ---
print("\nCopying training images...")
copy_images(train_df, 'train')

print("\nCopying validation images...")
copy_images(val_df, 'val')

print("\nCopying test images...")
copy_images(test_df, 'test')

# --- Save Split CSVs (Optional for Tracking) ---
train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
val_df.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)

# --- Summary ---
print("\n✅ Splitting complete!")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

