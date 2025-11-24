# This python fil will extract all the features from the images of a set
import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
        print(f"FeatureExtractor initialized with image path: {self.image_path}")

    def pre_process_image(self):
        img = cv2.imread(self.image_path)

        if img is None:
            raise ValueError(f"Image not found or unable to read: {self.image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if gray.mean() < 50 :
            _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_BINARY)
        else:
            _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        return mask, gray, img
    
    def getShapeDescriptor(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area/hull_area if hull_area>0 else 0

        features = {
            "area" : area,
            "perimeter" : perimeter,
            "bbox_width" :w,
            "bbox_height" : h,
            "aspect_ratio" : w/h if h>0 else 99,
            "circularity": 4*np.pi*area/(perimeter*perimeter) if perimeter>0 else 0,
            "solidity" : solidity,
            "extent" : area/(w*h) if h>0 else 0
        }
        return features, cnt
    
    def getColorDescriptor(self, img, gray, cnt):
        hog = cv2.HOGDescriptor()
        resized_img = cv2.resize(gray, (64, 128))
        hog_features = hog.compute(resized_img)

        features = {
            "hog": hog_features,
        }

        M = cv2.moments(cnt)
        hu = cv2.HuMoments(M).flatten()
        for i in range(7):
            if abs(hu[i]) < 1e-10:
                hu[i] = 1e-10
            features[f"hu_{i}"] = hu[i]
            features[f"hu_log_{i}"] = -np.sign(hu[i]) * np.log10(abs(hu[i]))
        
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        pixels = hsv[max==255]
        if len(pixels) > 10:
            mean = pixels.mean(axis=0)
            std = pixels.std(axis=0)
            features["hue_mean"] = mean[0]
            features["saturation_mean"] = mean[1]
            features["value_mean"] = mean[2]
            features["hue_std"] = std[0]
            features["sat_std"] = std[1]
            features["value_std"] = std[2]
        return features
