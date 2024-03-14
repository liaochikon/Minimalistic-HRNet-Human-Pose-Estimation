import torchvision.transforms as transforms
from dataset import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
from util.bbox_util import *
import torch.nn as nn
import torch

#model = HRNet(out_channels=21, base_channels=48)
#input = torch.randn(1, 3, 384, 288)
#output = model.forward(input)
#print(output.size())

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_annopath = "data\\annotations\\person_keypoints_wholebody_val.json"
val_imagepath = "data\\val2017"
val_dataset = COCOWholebody_BBox(val_annopath, val_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize]),
                                         image_height=288, image_width=384, heatmap_height=72, heatmap_width=96) 

print(len(val_dataset))

#for i in range(len(val_dataset)):
#    img = val_dataset.get_preprocessed_image(i)
#    targets = val_dataset.generate_heatmap_from_bbox(i)
#    
#    for cat, target in enumerate(targets):
#        labelsa, bboxsa = target_to_bboxs(target, cat, val_dataset.image_width, val_dataset.image_height)
#        bboxsa = xywh_to_xyxy(bboxsa)
#
#        labelsb, bboxsb = target_to_bboxs(target, cat, val_dataset.image_width, val_dataset.image_height, bbox_scale_ratio=2.5)
#        bboxsb = xywh_to_xyxy(bboxsb)
#
#        best_bboxs = sort_best_bboxs(bboxsa, bboxsb)

        #c = nn.MSELoss(reduction='mean')
        #print(best_bboxs)
        #print(c(torch.from_numpy(best_bboxs), torch.from_numpy(bboxsa)))

        #for l, b in zip(labelsa, bboxsa):
        #    if l == 0:
        #        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 1)
        #    if l == 1:
        #        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
        #    if l == 2:
        #        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
        #    if l == 3:
        #        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 1)
#
        #cv2.imshow("img", img)
        #cv2.waitKey(0) 
        