import streamlit as st
import cv2
import numpy as np
from image_segmentation import ImageSegmenter

st.title("Pill Detector")
st.write("This app is to demonstrate a prototype of the machine pill detector. Please follow the setup:")
st.html("<ul><li>Comlete White background</li> <li>Make sure the camera is close enough to the desk/surface</li> </ul>")
run = st.checkbox('Try Demo')
FRAME_WINDOW_main = st.image([])
FRAME_WINDOW_mask = st.image([])

camera=cv2.VideoCapture(0)
segmenter = ImageSegmenter()

while run:
    _, frame = camera.read()
    extracted_pills, bbox , thresh = segmenter.segment(frame)
    display_frame = frame.copy()

    for (x, y, w, h) in bbox:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(display_frame, 'Pill', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW_main.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    FRAME_WINDOW_mask.image(thresh)
else:
    st.write('Stopped')