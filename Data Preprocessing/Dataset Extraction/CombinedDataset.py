import os
import pandas as pd
import requests
import time
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

# === Paths ===
datasets = {
    'HAM': {
        'img_dirs': [
            r'C:\Users\shore\Desktop\APS360\Datasets\HAM10000\2\HAM10000_images_part_1',
            r'C:\Users\shore\Desktop\APS360\Datasets\HAM10000\2\HAM10000_images_part_2'
        ],
        'meta': r'C:\Users\shore\Desktop\APS360\Datasets\HAM10000\2\HAM10000_metadata.csv',
        'id_col': 'image_id',
        'label_col': 'dx'
    },
    'PAD': {
        'img_dirs': [
            r'C:\Users\shore\Desktop\APS360\Datasets\PAD-UFES-20\imgs_part_1\imgs_part_1',
            r'C:\Users\shore\Desktop\APS360\Datasets\PAD-UFES-20\imgs_part_2\imgs_part_2',
            r'C:\Users\shore\Desktop\APS360\Datasets\PAD-UFES-20\imgs_part_3\imgs_part_3'
        ],
        'meta': r'C:\Users\shore\Desktop\APS360\Datasets\PAD-UFES-20\metadata.csv',
        'id_col': 'img_id',
        'label_col': 'diagnostic'
    },
    'BCN': {
        'img_dirs': [r'C:\Users\shore\Desktop\APS360\Datasets\BCN20000_Images'],
        'meta': r'C:\Users\shore\Desktop\APS360\Datasets\bcn20000_metadata_2025-07-30.csv',
        'id_col': 'isic_id',
        'label_col': 'diagnosis_3'
    },
    'DERM12345': {
        'img_dirs': [r'C:\Users\shore\Desktop\APS360\Datasets\derm12345\images'],
        'meta': r'C:\Users\shore\Desktop\APS360\Datasets\derm12345.csv',
        'id_col': 'isic_id',
        'label_col': 'diagnosis_3'
    }
}

output_dir = r'C:\Users\shore\Desktop\APS360\Datasets\Combined'
os.makedirs(output_dir, exist_ok=True)

# === Label Mapping ===
def map_label(source, label_raw):
    if pd.isna(label_raw):
        return None

    label = label_raw.strip().lower()

    if source == 'HAM':
        if label == 'nv': return 'nevus'
        if label == 'mel': return 'melanoma'
        if label == 'bcc': return 'bcc'
        if label == 'bkl': return 'keratosis'
        if label == 'akiec': return 'actinic_keratosis'
        if label == 'vasc': return 'vascular_lesion'
        if label == 'df': return 'dermatofibroma'
        return None

    if source == 'PAD':
        if label == 'nev': return 'nevus'
        if label == 'mel': return 'melanoma'
        if label == 'bcc': return 'bcc'
        if label == 'ack': return 'actinic_keratosis'
        if label == 'sek': return 'keratosis'
        if label == 'scc': return 'scc'
        return None

    if source == 'BCN':
        if 'nevus' in label: return 'nevus'
        if 'melanoma' in label: return 'melanoma'
        if 'actinic keratosis' in label: return 'actinic_keratosis'
        if 'seborrheic keratosis' in label or 'keratosis' in label: return 'keratosis'
        if 'basal cell carcinoma' in label: return 'bcc'
        if 'squamous cell carcinoma' in label: return 'scc'
        if 'vascular' in label: return 'vascular_lesion'
        if 'lentigo' in label: return 'lentigo'
        if 'dermatofibroma' in label: return 'dermatofibroma'
        return None

    if source == 'DERM12345':
        if 'nevus' in label: return 'nevus'
        if 'melanoma' in label: return 'melanoma'
        if 'basal cell' in label or 'bcc' in label: return 'bcc'
        if 'squamous cell' in label or 'scc' in label: return 'scc'
        if 'actinic' in label: return 'actinic_keratosis'
        if 'seborrheic' in label or 'keratosis' in label: return 'keratosis'
        if 'dermatofibroma' in label: return 'dermatofibroma'
        if 'lentigo' in label: return 'lentigo'
        if 'vascular' in label or 'hemangioma' in label: return 'vascular_lesion'
        return None

# === Load and Process Datasets ===
all_records = []
missing_images = []
other_labels_counter = Counter()

for dataset_name, config in datasets.items():
    records = []
    df = pd.read_csv(config['meta'])

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Processing {dataset_name}'):
        img_base = str(row[config['id_col']]).strip()
        possible_exts = ['.jpg', '.jpeg', '.png']
        found = False
        for ext in possible_exts:
            img_name = f"{img_base}{ext}"
            for d in config['img_dirs']:
                src = os.path.join(d, img_name)
                if os.path.exists(src):
                    records.append({
                        'img_name': img_name,
                        'src': src,
                        'label_raw': str(row[config['label_col']]),
                        'source': dataset_name
                    })
                    found = True
                    break
            if found:
                break
        if not found:
            missing_images.append(img_base)

    # === Downsample Nevus Class in This Dataset ===
    nevus_records = [r for r in records if map_label(r['source'], r['label_raw']) == 'nevus']
    other_records = [r for r in records if map_label(r['source'], r['label_raw']) != 'nevus']

    if len(nevus_records) > 2000:
        nevus_sampled = pd.DataFrame(nevus_records).sample(n=2500, random_state=42).to_dict(orient='records')
    else:
        nevus_sampled = nevus_records

    all_records.extend(nevus_sampled + other_records)

# === Resize and Save Images ===
final = []
for rec in tqdm(all_records, desc='Resizing and Saving'):
    label = map_label(rec['source'], rec['label_raw'])
    if not label:
        other_labels_counter[rec['label_raw']] += 1
        continue

    src = rec['src']
    out = os.path.join(output_dir, rec['img_name'])

    try:
        img = Image.open(src).convert("RGB")
        img = img.resize((512, 512))
        img.save(out)
        final.append({'filename': rec['img_name'], 'label': label})
    except Exception as e:
        print(f"Error processing {src}: {e}")
        continue

# === Save Combined Labels CSV ===
df_final = pd.DataFrame(final)
label_csv = os.path.join(output_dir, 'combined_labels.csv')
df_final.to_csv(label_csv, index=False)
print(f"\n✅ Combined CSV saved to {label_csv}")
print(f"\n✅ Total Processed Images: {len(df_final)}")

# === Print Missing Images ===
print(f"\n❗ Total Missing Images: {len(missing_images)}")
if missing_images:
    print("Sample missing images:", missing_images[:5])

# === Print 'Other' Labels Summary ===
print(f"\n❗ Labels mapped to 'other' (excluded from dataset):")
for label, count in other_labels_counter.items():
    print(f"   {label}: {count} instances")

# === Visualize Class Distribution ===
plt.figure(figsize=(10, 5))
df_final['label'].value_counts().plot(kind='bar')
plt.title('Combined Dataset Class Distribution (512x512)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
