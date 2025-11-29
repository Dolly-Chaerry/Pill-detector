import os
import cv2
import numpy as np
from image_segmentation import ImageSegmenter
from feature_extractor import FeatureExtractor
from config import DATASET_UNCROPPED, EXPORT_CROPPED, DATASET_ROOT # do not forget to change this in config file for other classes
from pathlib import Path
from tqdm import tqdm
from sklearn import svm
import shutil
from sklearn.model_selection import train_test_split
'''
Prepare Dataset
- orgranize images to their classes
- segment images
- split dataset
- feature extraction and scaler fitting
'''
# IMAGE SEGMENTATION OF DATASET - only run this section once
category = ['capsule', 'tablet']
ctr = 0

print("STARTING IMAGE SEGMENTATION OF DATASET")
for cat in category:
    data = Path(DATASET_UNCROPPED + f"{cat}/")
    output = Path(EXPORT_CROPPED + f"{cat}/")
    output.mkdir(parents=True, exist_ok=True)

    paths = list(data.rglob("*.[jpg]*"))

    if not paths:
        print(f"No images found in {data}")
        continue
    
    print(f"Processing category: {cat.upper()} ({len(paths)} images)")
    for path in tqdm(paths, desc=cat, unit="img"):
        img_seg = ImageSegmenter(cv2.imread(path))
        extracted, bboxes = img_seg.segment()
        if extracted:
            cv2.imwrite(EXPORT_CROPPED + f"{cat}/" + f"/capsule{ctr}.png", extracted[0]) #change to class
        ctr+=1
print("Segmentation complete")

#SPLITTING DATA
print("SPLITTING DATASET")
sets = ["train","val","test"]
for split in sets:
    for cls in category:
        (Path(f"{DATASET_ROOT}/{split}/{cls}")).mkdir(parents=True, exist_ok=True)

for cls in category:
    paths = list(Path(f"{EXPORT_CROPPED}/{cls}").glob("*.png"))
    print(f"{cls}: {len(paths)} cropped pills")

    # First: separate test set (10%)
    train_val, test = train_test_split(paths, test_size=0.10, random_state=42)
    # Then: from remaining, take 15% as validation (~13.5% of total)
    train, val = train_test_split(train_val, test_size=0.15, random_state=42)

    for src_paths, split in [(train, "train"), (val, "val"), (test, "test")]:
        for p in src_paths:
            shutil.copy(p, Path(f"{DATASET_ROOT}/{split}/{cls}/{p.name}"))

print("\nFinal dataset ready in 'dataset/' with proper train/val/test split!")

#FEATURE EXTRACTION AND SCALING
# extractor = FeatureExtractor()
# extractor.fit_and_save('features.npz')

# train model
# data = np.load("features.npz")
# X = data['features']
# y = data['labels']


# test performance

# save model

