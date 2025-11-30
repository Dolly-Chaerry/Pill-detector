import cv2
import numpy as np
from config import RESIZE

#need to resize image before inference resize to 244,244 pixels. 
class ImageSegmenter:
    def __init__(self):
        print("Initiated Segmenter")
        pass
        
    def segment(self, img):
        hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
        s = hsv[:,:,1]
        s_blur = cv2.GaussianBlur(s, (5,5), 0)
        _, thresh1 = cv2.threshold(s_blur, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh2 = cv2.threshold(s_blur, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.bitwise_and(thresh1,thresh2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(thresh.shape, dtype=np.uint8)

        min_area = 300

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
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        mask = (mask > 0).astype(np.uint8) * 255

        extracted_pills = []
        bboxes = []

        for cnt in valid_contours:
            # Individual mask for this pill only
            single_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(single_mask, [cnt], -1, 255, thickness=cv2.FILLED)

            x, y, w, h = cv2.boundingRect(cnt)

            # Crop just this pill region (much faster than full-image operations)
            pill_bgr_crop = img[y:y+h, x:x+w]
            mask_crop = single_mask[y:y+h, x:x+w]

            if pill_bgr_crop.size == 0:
                continue

            # Convert to BGRA and apply exact mask
            pill_bgra = cv2.cvtColor(pill_bgr_crop, cv2.COLOR_BGR2BGRA)
            pill_bgra[:, :, 3] = mask_crop

            # === Resize with aspect ratio preserved + transparent padding ===
            oh, ow = pill_bgra.shape[:2]
            scale = RESIZE / max(oh, ow)
            new_h = int(oh * scale + 0.5)
            new_w = int(ow * scale + 0.5)

            resized = cv2.resize(pill_bgra, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            padded = np.zeros((RESIZE, RESIZE, 4), dtype=np.uint8)  # black + transparent
            top = (RESIZE - new_h) // 2
            left = (RESIZE - new_w) // 2
            padded[top:top+new_h, left:left+new_w] = resized

            extracted_pills.append(padded)
            bboxes.append((x, y, w, h))

        return extracted_pills, bboxes , thresh