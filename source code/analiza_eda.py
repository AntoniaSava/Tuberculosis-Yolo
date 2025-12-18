import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import os

META_CSV = 'data/MetaData.csv'
MASK_DIR = 'data/mask'

df = pd.read_csv(META_CSV)
plt.figure(figsize=(6, 4))
sns.countplot(x='ptb', data=df, palette='viridis')
plt.title('Class Distribution: 0=Healthy vs 1=Tuberculosis')
plt.xlabel('Diagnosis (PTB)')
plt.ylabel('Image Count')
plt.savefig('eda_class_balance.png')
plt.close()

mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
lesion_areas = []

for mask_file in mask_files:
    path = os.path.join(MASK_DIR, mask_file)
    mask = cv2.imread(path, 0)
    if mask is not None:
        area = cv2.countNonZero(mask)
        if area > 0:
            lesion_areas.append(area)

plt.figure(figsize=(8, 4))
sns.histplot(lesion_areas, bins=30, color='salmon', kde=True)
plt.title('Lesion Size Distribution (Pixel Area)')
plt.xlabel('Lesion Area (pixels)')
plt.ylabel('Frequency')
plt.savefig('eda_lesion_size.png')
plt.close()

print("Graficele au fost generate: eda_class_balance.png si eda_lesion_size.png")