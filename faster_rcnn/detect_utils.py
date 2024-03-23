import torchvision.transforms as transforms
import cv2
import numpy as np
from faster_rcnn.coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    outputs = model(image)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    return boxes, pred_classes, outputs[0]['labels']

def draw_boxes(boxes, classes, labels, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA)
    return image

def person_boxes(boxes, labels, image):
    p_boxes = []
    for b, l in zip(boxes, labels):
        if l != 1:
            continue
        p_boxes.append(b)
        
    return image, p_boxes