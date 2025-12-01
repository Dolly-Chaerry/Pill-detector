import streamlit as st
import cv2
import numpy as np
from image_segmentation import ImageSegmenter
from feature_extractor import FeatureExtractor
import joblib
from PIL import Image

segmenter = ImageSegmenter()
extractor = FeatureExtractor()
svm_model = joblib.load("svm_model.joblib")

st.title("Pill Detector")
st.write("This app is to demonstrate a prototype of the machine pill detector. Please follow the setup:")
st.html("<ul><li>Comlete White background</li> <li>Make sure the camera is close enough to the desk/surface</li> <li>Angle the camera in top view </li> </ul>")

img = st.camera_input("Take a photo of the pills")

if img is not None:
    img = cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8),cv2.IMREAD_COLOR)
    
    pills, bbox, thresh = segmenter.segment(img)
    display_frame = img.copy()
    predictions =[]
    capsule = 0
    tablet = 0
    for i, pill in enumerate(pills):
        features = extractor.get_features(pill)
        predict = svm_model.predict([features])
        pred_label = "capsule" if predict == 0 else "tablet"
        predictions.append(pred_label)
        if predict == 0: 
            capsule+=1
        else: tablet+=1

        if i < len(bbox):
            x, y, w, h = bbox[i]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    
            label_text = pred_label

            cv2.putText(
                        display_frame,
                        label_text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                )
        overlay = display_frame.copy()
        h, w = display_frame.shape[:2]

        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

        cv2.putText(display_frame, f"CAPSULES: {capsule}", (20, 85),cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(display_frame, f"TABLETS: {tablet}", (400, 85),cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 100, 0), 3)
    st.image(thresh, "MASK")
    st.image(display_frame, "OUTPUT")