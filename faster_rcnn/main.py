import torchvision
import numpy as np
import torch
import cv2
import detect_utils

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=800)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval().to(device)

while True:
    ret, frame = cap.read()
    boxes, classes, labels = detect_utils.predict(frame, model, device, 0.8)
    frame = detect_utils.draw_boxes(boxes, classes, labels, frame)
    cv2.imshow('Image', frame)
    cv2.waitKey(1)