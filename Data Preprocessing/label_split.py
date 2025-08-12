import pandas as pd
import matplotlib.pyplot as plt

# Path to your combined labels CSV
csv_path = r'C:\Users\shore\Desktop\APS360\Datasets\Combined\combined_labels.csv'

# Load the dataset
df = pd.read_csv(csv_path)

# Count instances of each label
class_distribution = df['label'].value_counts()

# Display result
print("🔢 Class Distribution in Combined Dataset:")
print(class_distribution)

# Optional: Save to CSV
output_path = csv_path.replace('combined_labels.csv', 'class_distribution.csv')
class_distribution.to_csv(output_path, header=['count'])
print(f"\n✅ Class distribution saved to: {output_path}")

# ===== Plot distribution =====
plt.figure(figsize=(10, 6))
class_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Class Distribution", fontsize=16)
plt.xlabel("Class", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
