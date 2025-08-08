import pandas as pd
import os
import requests
import time

# Load BCN20000 metadata
metadata_csv = r'C:\Users\shore\Desktop\APS360\Datasets\bcn20000_metadata_2025-07-30.csv'
save_dir = r'C:\Users\shore\Desktop\APS360\Datasets\BCN20000_Images'
os.makedirs(save_dir, exist_ok=True)

# Load metadata
metadata = pd.read_csv(metadata_csv).reset_index(drop=True)

# Prepare labels file in append mode
label_file_path = os.path.join(save_dir, 'labels.csv')
if not os.path.exists(label_file_path):
    with open(label_file_path, 'w') as label_file:
        label_file.write('filename,label\n')

# Set starting index based on where you left off
#start_index = metadata.index[metadata['isic_id'] == 'ISIC_0069998'][0]

# Download remaining images
for idx in range(len(metadata)):
    row = metadata.iloc[idx]
    isic_id = row['isic_id']
    label = row['diagnosis_1']  # Adjust if label column is named differently

    filename = f"{isic_id}.jpg"
    image_url = f"https://isic-archive.s3.amazonaws.com/images/{filename}"
    save_path = os.path.join(save_dir, filename)

    if not os.path.exists(save_path):
        try:
            print(f"Downloading {filename}...")
            img_data = requests.get(image_url).content

            with open(save_path, 'wb') as handler:
                handler.write(img_data)

            with open(label_file_path, 'a') as label_file:
                label_file.write(f"{filename},{label}\n")

            if (idx + 1) % 100 == 0:
                print(f"{idx + 1} images processed...")

        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            time.sleep(5)
            continue
    else:
        print(f"{filename} already exists. Skipping.")

print("✅ Remaining BCN20000 images downloaded successfully!")



