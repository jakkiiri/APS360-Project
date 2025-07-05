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
    'HAM10000': {
        'img_dirs': [
            r'C:\Users\shore\Desktop\APS360\Datasets\HAM10000\HAM10000_images_part_1',
            r'C:\Users\shore\Desktop\APS360\Datasets\HAM10000\HAM10000_images_part_2'
        ],
        'meta': r'C:\Users\shore\Desktop\APS360\Datasets\HAM10000\HAM10000_metadata.csv',
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
        'meta': r'C:\Users\shore\Desktop\APS360\Datasets\BCN20000_metadata.csv',
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
        code = label_raw.strip().lower()
        if code == 'nv': return 'nevus'
        if code == 'mel': return 'melanoma'
        if code == 'bcc': return 'bcc'
        if code == 'bkl': return 'seborrheic_keratosis'
        if code == 'akiec': return 'actinic_keratosis'
        if code == 'vasc': return 'vascular_lesion'
        if code == 'df': return 'dermatofibroma'
        return None

    if source == 'PAD':
        code = label_raw.strip().upper()
        if code == 'NEV': return 'nevus'
        if code == 'MEL': return 'melanoma'
        if code == 'BCC': return 'bcc'
        if code == 'ACK': return 'actinic_keratosis'
        if code == 'SEK': return 'seborrheic_keratosis'
        if code == 'SCC': return 'scc'
        return None

    if source == 'BCN':
        if 'nevus' in label: return 'nevus'
        if 'melanoma' in label: return 'melanoma'
        if 'actinic keratosis' in label: return 'actinic_keratosis'
        if 'seborrheic keratosis' in label: return 'seborrheic_keratosis'
        if 'keratosis' in label: return 'other_keratosis'
        if 'basal cell carcinoma' in label: return 'bcc'
        if 'squamous cell carcinoma' in label: return 'scc'
        if 'vascular' in label: return 'vascular_lesion'
        if 'lentigo' in label: return 'lentigo'
        if 'dermatofibroma' in label: return 'dermatofibroma'
        return None

# === Load and Process Datasets ===
records = []
missing_images = []
other_labels_counter = Counter()

# HAM10000
df_ham = pd.read_csv(datasets['HAM10000']['meta'])
for idx, row in tqdm(df_ham.iterrows(), total=len(df_ham), desc='Processing HAM10000'):
    img_name = f"{row[datasets['HAM10000']['id_col']]}.jpg"
    found = False
    for d in datasets['HAM10000']['img_dirs']:
        src = os.path.join(d, img_name)
        if os.path.exists(src):
            records.append({'img_name': img_name, 'src': src, 'label_raw': row[datasets['HAM10000']['label_col']], 'source': 'HAM'})
            found = True
            break
    if not found:
        missing_images.append(img_name)

# PAD-UFES
df_pad = pd.read_csv(datasets['PAD']['meta'])
for idx, row in tqdm(df_pad.iterrows(), total=len(df_pad), desc='Processing PAD-UFES'):
    img_name = row[datasets['PAD']['id_col']]
    found = False
    for d in datasets['PAD']['img_dirs']:
        src = os.path.join(d, img_name)
        if os.path.exists(src):
            records.append({'img_name': img_name, 'src': src, 'label_raw': row[datasets['PAD']['label_col']], 'source': 'PAD'})
            found = True
            break
    if not found:
        missing_images.append(img_name)

# BCN20000
df_bcn = pd.read_csv(datasets['BCN']['meta'])
for idx, row in tqdm(df_bcn.iterrows(), total=len(df_bcn), desc='Processing BCN20000'):
    img_name = f"{row[datasets['BCN']['id_col']]}.jpg"
    src = os.path.join(datasets['BCN']['img_dirs'][0], img_name)
    if os.path.exists(src):
        records.append({'img_name': img_name, 'src': src, 'label_raw': row[datasets['BCN']['label_col']], 'source': 'BCN'})
    else:
        missing_images.append(img_name)

# === Resize and Save Images ===
final = []
for rec in tqdm(records, desc='Resizing and Saving'):
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
