import cv2
import numpy as np

class ImageSegmenter:
    def __init__(self):
        print("ImageSegmenter initialized.")
    def segment_image(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        s_blur = cv2.GaussianBlur(s, (5, 5), 0)
        _, thresh1 = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh2 = cv2.threshold(s_blur, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.bitwise_or(thresh1, thresh2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(thresh.shape, dtype=np.uint8)

        min_area = 400

        valid_contours =[]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.3:
                        valid_contours.append(cnt)
                        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        print(len(valid_contours))

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        mask = (mask > 0).astype(np.uint8) * 255

        