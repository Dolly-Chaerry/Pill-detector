from feature_extraction import FeatureExtractor
from config import DATASET_PATH, DATA_CSV_PATH
from pathlib import Path
import cv2
import pandas as pd

#Check if features have been extracted, else proceed to feature extraction
data_path = Path(DATA_CSV_PATH)
if data_path.exists():
    print("Data file exists")
    #delete data if you want to re-extract features
else:
    print("Data file does not exist. Running extraction...")
    features_list = []

    for file in Path(DATASET_PATH).iterdir():
        if file.is_file() and file.suffix in [".jpg", ".png", ".jpeg"]:
            print(f"Processing file: {DATASET_PATH}/{file.name}")
            
            #Feature Extraction
            img = cv2.imread(f"{DATASET_PATH}/{file.name}")
            if img is None:
                print(f"Failed to load image: {DATASET_PATH}/{file.name}")
                continue
            extractor = FeatureExtractor(f"{DATASET_PATH}/{file.name}")
            mask, gray, img = extractor.pre_process_image()
            shape_features, cnt = extractor.getShapeDescriptor(mask)
            color_features = extractor.getColorDescriptor(img, gray, cnt)
            features = {**shape_features, **color_features}
            # print(f"Extracted features for {file.name}: {features}")
            features_list.append(features)

    print(f"Total images processed: {len(features_list)}")

    # Save features to CSV
    df = pd.DataFrame(features_list)
    df.to_csv(DATA_CSV_PATH, index=False)

#Training Model


