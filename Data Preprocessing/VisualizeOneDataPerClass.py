import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# --- Root dataset path (adjust this) ---
dataset_dir = r'C:\Users\shore\Desktop\APS360\Datasets\DataSplit\train'  # or 'test' or full dataset

# --- Get class folders ---
class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
class_dirs.sort()  # for consistency

# --- Plot settings ---
num_classes = len(class_dirs)
cols = min(num_classes, 5)
rows = (num_classes + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))
axes = axes.flatten()

for idx, class_name in enumerate(class_dirs):
    class_path = os.path.join(dataset_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not images:
        print(f"No images found in {class_path}")
        continue

    sample_img = random.choice(images)  # randomly pick one
    img_path = os.path.join(class_path, sample_img)
    img = mpimg.imread(img_path)

    axes[idx].imshow(img)
    axes[idx].set_title(class_name, fontsize=10)
    axes[idx].axis('off')

# --- Hide unused subplots ---
for ax in axes[num_classes:]:
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Reserve 10% top margin
plt.suptitle("One Sample Image per Class", fontsize=14, y=1.02)
plt.show()
