import cv2
import numpy as np
import os
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn import preprocessing
from config import TRAIN_ROOT

class FeatureExtractor:
    def __init__(self, scaler_path="feature_scaler.pkl"):
        self.scaler_path = scaler_path
        self.scaler = None
        self.is_fitted = False
        self._load_or_create_scaler()

    def _load_or_create_scaler(self):
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            self.is_fitted = True
            print(f"Scaler loaded from {self.scaler_path}")
        else:
            self.scaler = preprocessing.StandardScaler()
            print("No scaler found → new one created (will fit during training)")

    def _composite_bgra(self, img):
        """Convert BGRA → BGR on black background"""
        if img.shape[2] == 4:
            bgr = img[:, :, :3].astype(float)
            alpha = img[:, :, 3:].astype(float) / 255.0
            bgr = bgr * alpha
            return np.clip(bgr, 0, 255).astype(np.uint8)
        return img[..., :3] if img.shape[2] == 4 else img

    def extract_features(self, input_data):
        """
        Input: either file path (str/Path) or numpy image array (H,W,3 or H,W,4)
        Output: raw feature vector (float32)
        """
        # Load image if path is given
        if isinstance(input_data, (str, Path)):
            img = cv2.imread(str(input_data), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Cannot read image: {input_data}")
        else:
            img = input_data

        bgr = self._composite_bgra(img)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # === Color Features (18) ===
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        color_feats = []
        for plane in [hsv, lab]:
            for i in range(3):
                ch = plane[:, :, i].ravel()
                ch = ch[ch > 10]
                if len(ch) == 0:
                    color_feats.extend([0.0, 0.0, 0.0])
                else:
                    mean, std = ch.mean(), ch.std()
                    skew = np.mean((ch - mean)**3) / (std**3 + 1e-8)
                    color_feats.extend([mean, std, skew])

        # === Shape Features (18) ===
        mask = (gray > 10).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            shape_feats = np.zeros(18)
        else:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            extent = area / (w * h) if w * h > 0 else 0
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
            equiv_diam = np.sqrt(4 * area / np.pi) if area > 0 else 0
            moments = cv2.moments(cnt)
            hu = cv2.HuMoments(moments).flatten()
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

            base_shape = [area, perimeter, circularity, aspect, extent,
                          solidity, equiv_diam, w, h]
            shape_feats = np.concatenate([base_shape, hu])

        # === Edge + Simple HOG (9) ===
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hist, _ = np.histogram(ang[mag > 10], bins=8, range=(0, 360), density=True)
        edge_feats = np.concatenate([[edge_density], hist])

        # Final raw feature vector
        features = np.concatenate([color_feats, shape_feats, edge_feats])
        return features.astype(np.float32)


    def fit_and_save(self, save_path="classical_features.npz"):
        features_list = []
        labels_list = []
        class_to_idx = {"capsules": 0, "tablet": 1}

        print("Extracting features from training set...")
        for class_name, label in class_to_idx.items():
            class_dir = Path(TRAIN_ROOT) / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found!")
                continue
            paths = list(class_dir.rglob("*.png"))
            print(f"{class_name}: {len(paths)} images")

            for p in tqdm(paths, desc=class_name):
                feats = self.extract_features(p)
                features_list.append(feats)
                labels_list.append(label)

        X = np.array(features_list)
        y = np.array(labels_list)

        # Fit scaler
        self.scaler.fit(X)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Scaler saved → {self.scaler_path}")
        self.is_fitted = True

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Save
        np.savez(save_path, features=X_scaled, labels=y)
        print(f"Training data saved → {save_path} | Shape: {X_scaled.shape}")

    def get_features(self, image_path_or_array):
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted! Run fit_and_save() first.")

        raw_feats = self.extract_features(image_path_or_array)
        scaled_feats = self.scaler.transform(raw_feats.reshape(1, -1))
        return scaled_feats.flatten()  # shape: (58,)
