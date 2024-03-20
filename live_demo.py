import torch
import torchvision.transforms as transforms
from model import HRNet
from dataset import COCOWholebody_BBox
from util.bbox_util import *
import numpy as np
import cv2

def get_affine(image_size, image_width, image_height):
    center = (image_size[1] / 2, image_size[0] / 2)
    scale = max([image_size[1] / image_width, image_size[0] / image_height])
    bbox_topleft = (center[0] - image_width / 2 * scale, center[1] - image_height / 2 * scale)
    bbox = [bbox_topleft[0], bbox_topleft[1], image_width * scale, image_height * scale]

    p1 = np.float32([[int(bbox[0]), int(bbox[1])],[int(bbox[0] + bbox[2]), int(bbox[1])],[int(bbox[0]), int(bbox[1] + bbox[3])]])
    p2 = np.float32([[0, 0],[image_width, 0],[0, image_height]])
    M = cv2.getAffineTransform(p1, p2)
    return M

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

image_width = 384
image_height =288
M = get_affine(frame.shape, 384, 288)

device = "cuda"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
T = transforms.Compose([transforms.ToTensor(), normalize])

model = HRNet(out_channels=1)
model.load_state_dict(torch.load("weight/best_loss.pth"))

model.eval()
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        image_preprocess = cv2.warpAffine(frame, M, (384, 288))
        image_transformed = T(image_preprocess)

        outputs = model(image_transformed.reshape((1, image_transformed.size(0),  image_transformed.size(1), image_transformed.size(2))))

        pred_list = []
        cat_list = []

        for cat, o in enumerate(outputs[0]):
            labels_t, bboxs_t = target_to_bboxs(o, cat, image_width, image_height, lower_thesh=150, bbox_scale_ratio=2)
            bboxs_t = xywh_to_xyxy(bboxs_t)
            pred_list.extend(bboxs_t)
            cat_list.extend(labels_t)
        

        for l, b in zip(cat_list, pred_list):
            if len(b) == 0:
                continue
            if l == 0:
                cv2.rectangle(image_preprocess, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 1)
            #if l == 1:
            #    cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 1)
            #if l == 2:
            #    cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            #if l == 3:
            #    cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 0), 1)

        cv2.imshow("img", image_preprocess)
        cv2.waitKey(1) 

        #if (i + 1) % print_freq == 0:
        #    get_output_result(val_dataset, outputs, targets, data_idx, "val", 0, i + 1)