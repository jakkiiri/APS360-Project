import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

# --- Read Distribution ---
def get_class_distribution(split_name):
    split_path = os.path.join(split_dir, split_name)
    class_counts = {label: 0 for label in label_mapping.keys()}
    for label in os.listdir(split_path):
        label_dir = os.path.join(split_path, label)
        if os.path.isdir(label_dir):
            num_images = len(os.listdir(label_dir))
            class_counts[label] += num_images
    return class_counts

# --- Setup ---
splits = ['train', 'val', 'test']
all_distributions = {split: get_class_distribution(split) for split in splits}
labels = list(label_mapping.keys())
colors = plt.cm.tab10.colors + plt.cm.Pastel1.colors
colors = colors[:len(labels)]
legend_handles = [Patch(facecolor=colors[i], label=labels[i]) for i in range(len(labels))]

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(20, 8))  # Taller to make room for legend

for i, split in enumerate(splits):
    counts = all_distributions[split]
    sizes = [counts[label] for label in labels]

    wedges, texts = axes[i].pie(
        sizes,
        colors=colors,
        startangle=140,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        labels=None,
        autopct=None,
        pctdistance=0.75
    )

    total = sum(sizes)
    for j, (wedge, size) in enumerate(zip(wedges, sizes)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = wedge.r * 1.2 * np.cos(np.radians(angle))
        y = wedge.r * 1.2 * np.sin(np.radians(angle))
        percentage = size / total * 100
        if percentage > 1:
            axes[i].text(x, y, f"{percentage:.1f}%", ha='center', va='center', fontsize=8)
        else:
            axes[i].text(x * 1.2, y * 1.2, f"{percentage:.1f}%", ha='center', va='center', fontsize=7)

    axes[i].set_title(f"{split.capitalize()} Split", fontsize=12)
    axes[i].axis('equal')

# --- Shared Legend Below ---
legend = fig.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=5,
    fontsize=9,
    bbox_to_anchor=(0.5, 0.15)
)

plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave extra space at the bottom
plt.show()

