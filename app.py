import streamlit as st
import cv2
import numpy as np
from image_segmentation import ImageSegmenter
st.title("Pill Detector")

# Toggle
enable = st.checkbox("Enable live camera", value=True)

if enable:
    # This gives you a LIVE webcam feed inside Streamlit!
    camera = st.camera_input(
        "Live Camera",
        key="live_cam",     # this is the important part
        label_visibility="collapsed"
    )

    # camera is None until the stream starts
    if camera is not None:
        # Convert the frame (it's already a PIL Image or bytes)
        frame = np.array(camera)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        segmenter = ImageSegmenter(frame)
        extracted, bboxes = segmenter.segment()

        #draw bounding boxes on frame
        display_frame = frame.copy()

        for (x, y, w, h) in bboxes:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(display_frame, 'Pill', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Optional: Show number of detected pills
        st.write(f"**Detected {len(bboxes)} pill(s)**")

        # Convert back to RGB for Streamlit display
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # Show the result with bounding boxes
        st.image(display_frame_rgb, caption="Detected Pills", use_column_width=True)
else:
    st.write("Camera is disabled")