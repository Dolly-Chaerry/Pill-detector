# Segment all the images in the data set and save them into their folder
# filename should be their class and index
import os
import cv2
from image_segmentation import ImageSegmenter
from feature_extractor import FeatureExtractor
from config import DATASET_UNCROPPED, EXPORT_CROPPED # do not forget to change this in config file for other classes
from pathlib import Path

#IMAGE SEGMENTATION OF DATASET - only run this section once
# data = Path(DATASET_UNCROPPED)
# output = Path(EXPORT_CROPPED)
# output.mkdir(parents=True, exist_ok=True)

# paths = list(data.rglob("*.[jpg]*"))

# ctr = 0
# for path in paths:
#     print("Segmenting :", path)
#     img_seg = ImageSegmenter(cv2.imread(path))
#     extracted, bboxes = img_seg.segment()
#     if extracted:
#         cv2.imwrite(EXPORT_PATH + f"/tablet{ctr}.png", extracted[0]) #change to class
#     print("Saving Complete")
#     ctr+=1
#     # if ctr == 10: break
# print("Segmentation complete")

#FEATURE EXTRACTION AND SCALING
extractor = FeatureExtractor()

extractor.fit_and_save('features.npz')