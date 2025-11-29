import os
import cv2
import numpy as np
from image_segmentation import ImageSegmenter
from feature_extractor import FeatureExtractor
from config import DATASET_UNCROPPED, EXPORT_CROPPED # do not forget to change this in config file for other classes
from pathlib import Path
from tqdm import tqdm
from sklearn import svm
'''
Prepare Dataset
- orgranize images to their classes
- segment uncropped train and validation images
- feature extraction and scaler fitting
'''
# IMAGE SEGMENTATION OF DATASET - only run this section once
category = ['capsule', 'tablet']
ctr = 0
#Training dataset
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

#FEATURE EXTRACTION AND SCALING
# extractor = FeatureExtractor()
# extractor.fit_and_save('features.npz')

# train model
# data = np.load("features.npz")
# X = data['features']
# y = data['labels']


# test performance

# save model

