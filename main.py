import os
import cv2
import numpy as np
from image_segmentation import ImageSegmenter
from feature_extractor import FeatureExtractor
from config import DATASET_UNCROPPED, EXPORT_CROPPED, DATASET_ROOT # do not forget to change this in config file for other classes
from pathlib import Path
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
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
segmenter = ImageSegmenter()
masks = []

# print("STARTING IMAGE SEGMENTATION OF DATASET")
# for cat in category:
#     data = Path(DATASET_UNCROPPED + f"{cat}/")
#     output = Path(EXPORT_CROPPED + f"{cat}/")
#     output.mkdir(parents=True, exist_ok=True)

#     paths = list(data.rglob("*.[jpg]*"))

#     if not paths:
#         print(f"No images found in {data}")
#         continue
    
#     print(f"Processing category: {cat.upper()} ({len(paths)} images)")
#     for path in tqdm(paths, desc=cat, unit="img"):
#         extracted, bboxes, _ = segmenter.segment(cv2.imread(path))
#         if extracted:
#             cv2.imwrite(EXPORT_CROPPED + f"{cat}/" + f"/{cat}{ctr}.png", extracted[0]) #change to class
#         ctr+=1
#     ctr=0
# print("Segmentation complete")

# # #SPLITTING DATA
# print("SPLITTING DATASET")
# sets = ["train","val","test"]
# for split in sets:
#     for cls in category:
#         (Path(f"{DATASET_ROOT}/{split}/{cls}")).mkdir(parents=True, exist_ok=True)

# for cls in category:
#     paths = list(Path(f"{EXPORT_CROPPED}/{cls}").glob("*.png"))
#     print(f"{cls}: {len(paths)} cropped pills")

#     # First: separate test set (10%)
#     train_val, test = train_test_split(paths, test_size=0.10, random_state=42)
#     # Then: from remaining, take 15% as validation (~13.5% of total)
#     train, val = train_test_split(train_val, test_size=0.15, random_state=42)

#     for src_paths, split in [(train, "train"), (val, "val"), (test, "test")]:
#         for p in src_paths:
#             shutil.copy(p, Path(f"{DATASET_ROOT}/{split}/{cls}/{p.name}"))

# print("\nFinal dataset ready in 'dataset/' with proper train/val/test split!")

# # FEATURE EXTRACTION AND SCALING
# print("Extracting train features + fitting scaler...")
# extractor = FeatureExtractor()                     # creates & fits scaler inside
# extractor.fit_and_save(save_path="train_features.npz")
# print("Training features + scaler ready!\n")

# val_extractor = FeatureExtractor()   # automatically loads the saved scaler
# val_features = []
# val_labels = []

# print("Extracting validation features...")
# for cls in category:
#     val_dir = Path(DATASET_ROOT) / "val" / cls
#     if not val_dir.exists():
#         print(f"Warning: {val_dir} not found!")
#         continue
        
#     paths = list(val_dir.glob("*.png"))
#     label = 0 if cls == "capsules" else 1
    
#     for p in paths:
#         feats = val_extractor.get_features(str(p))
#         val_features.append(feats)
#         val_labels.append(label)

# val_features = np.array(val_features)
# val_labels = np.array(val_labels)
# np.savez("val_features.npz", features=val_features, labels=val_labels)
# print(f"Validation features saved → {val_features.shape}\n")

# test_extractor = FeatureExtractor()
# test_features = []
# test_labels = []

# print("Extracting test features...")
# for cls in category:
#     test_dir = Path(DATASET_ROOT) / "test" / cls
#     if not test_dir.exists():
#         print(f"Warning: {test_dir} not found!")
#         continue
        
#     paths = list(test_dir.glob("*.png"))
#     label = 0 if cls == "capsules" else 1
    
#     for p in paths:
#         feats = test_extractor.get_features(str(p))
#         test_features.append(feats)
#         test_labels.append(label)

# test_features = np.array(test_features)
# test_labels = np.array(test_labels)
# np.savez("test_features.npz", features=test_features, labels=test_labels)
# print(f"Test features saved → {test_features.shape}")


# train model
train = np.load("train_features.npz")
val   = np.load("val_features.npz")
test  = np.load("test_features.npz")

X_train, y_train = train["features"], train["labels"]
X_val,   y_val   = val["features"],   val["labels"]
X_test,  y_test  = test["features"],  test["labels"]

best_score = 0
best_C = None

print("Tuning SVM hyperparameter C...")
for C in [0.1, 1, 5, 10, 25, 50, 100, 500]:
    svm = SVC(C=C, kernel='rbf', class_weight='balanced')
    svm.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, svm.predict(X_val))
    
    print(f"   C={C:>5} → val accuracy: {val_acc:.4f}")
    if val_acc > best_score:
        best_score = val_acc
        best_C = C

print(f"\nBest C = {best_C} (validation accuracy = {best_score:.4f})")

# FINAL MODEL: retrain on train + validation
final_svm = SVC(C=best_C, kernel='rbf', class_weight='balanced')
final_svm.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

test_acc = accuracy_score(y_test, final_svm.predict(X_test))
print(f"REAL FINAL TEST ACCURACY: {test_acc:.4f} → This is your production performance!\n")

model_path = "svm_model.joblib"
joblib.dump(final_svm, model_path)
print(f"Model saved → {model_path}")