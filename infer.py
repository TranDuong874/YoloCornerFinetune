from ultralytics import YOLO

img_path = r'real_and_synthetic_corner_dataset\images\0018.jpg'
model = YOLO('pretrained_model/yolov8s-pose.pt')
results = model(img_path)
print(results)

import cv2
img = cv2.imread(img_path)
