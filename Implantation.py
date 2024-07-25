import cv2
import numpy as np
import torch
from pathlib import Path

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the image
img = cv2.imread('ImageSaveLocation')


# Perform object detection on the image
results = model(img)

# Get mask of person pixels
mask = np.zeros_like(img)
for result in results.pred:
    class_id = result[:, -1]
    mask[class_id == 0] = 255

# Perform image segmentation
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
mask = cv2.bitwise_or(mask, mask, mask=thresh)

# Cut out person from background
person = cv2.bitwise_and(img, mask)

# Replace background
new_bg_path = Path('ImageSaveLocation')
new_bg = cv2.imread(str(new_bg_path))
new_bg = cv2.resize(new_bg, (img.shape[1], img.shape[0]))
# Invert mask to get background pixels
inv_mask = cv2.bitwise_not(mask)

# Get background from new background image
bg = cv2.bitwise_and(new_bg, inv_mask)

# Combine person and new background
new_img = cv2.add(person, bg)

# Display result
cv2.imwrite('ImageSaveLocation', new_img)


