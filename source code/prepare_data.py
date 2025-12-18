import os
import cv2
import numpy as np
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_DIR = 'data/image'
MASK_DIR = 'data/mask'
OUT_DIR = 'data/yolo'
META_CSV = 'data/MetaData.csv'  


df_meta = pd.read_csv(META_CSV)

df_meta['image_name'] = df_meta['id'].astype(str) + '.png' 

img_to_label = {}
for idx, row in df_meta.iterrows():
    ptb = row['ptb']
    if ptb == 0:
        label = 1  # sanatos
    else:
        label = 0 # nesanatos
    img_to_label[row['image_name']] = label


for split in ['train', 'val']:
    os.makedirs(f'{OUT_DIR}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/labels/{split}', exist_ok=True)


images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)


def convert_mask_to_yolo(mask_path, label_path, label):
    mask = cv2.imread(mask_path, 0)
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape

    with open(label_path, 'w') as f:
        for contour in contours:
            if len(contour) < 6:
                continue
            points = [(pt[0][0] / w, pt[0][1] / h) for pt in contour]
            flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
            f.write(f"{label} {flat}\n")  


for split, imgs in [('train', train_imgs), ('val', val_imgs)]:
    for img in imgs:
        mask = img.replace('.jpg', '.png').replace('.jpeg', '.png')

    
        label = img_to_label.get(img, 0)  

     
        shutil.copy(os.path.join(IMG_DIR, img), os.path.join(OUT_DIR, f'images/{split}/{img}'))

        
        convert_mask_to_yolo(
            os.path.join(MASK_DIR, mask),
            os.path.join(OUT_DIR, f'labels/{split}/{img.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")}'),
            label
        )

print("Datele au fost convertite și structurate pentru YOLOv8.")
