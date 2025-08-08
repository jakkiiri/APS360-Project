import pandas as pd

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
