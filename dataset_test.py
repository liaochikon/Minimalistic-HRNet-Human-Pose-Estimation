import torchvision.transforms as transforms
from dataset import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn as nn

#model = HRNet(out_channels=21, base_channels=48)
#input = torch.randn(1, 3, 384, 288)
#output = model.forward(input)
#print(output.size())

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_annopath = "data\\halpe_fullbody\\annotations\\halpe_train_v1.json"
val_imagepath = "data\\halpe_fullbody\\train2015"
val_dataset = Halpe_Fullbody(val_annopath, val_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize])) 

#val_annopath = "data\\coco_wholebody\\annotations\\coco_wholebody_val_v1.0.json"
#val_imagepath = "data\\coco_wholebody\\val2017"
#val_dataset = COCOWholebody_BodyWithFeet(val_annopath, val_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize])) 

print(len(val_dataset))
for i in range(len(val_dataset)):
    img = val_dataset.get_preprocessed_image(i)
    joints = val_dataset.get_preprocessed_joints(i)
    joint_vis = val_dataset.get_preprocessed_joint_vis(i)

    for jt, (j, jv) in enumerate(zip(joints, joint_vis)):
        if jv[0] > 0:
            cv2.putText(img, str(jt), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.circle(img,(int(j[0]), int(j[1])), 1, (0, 0, 255), 1)
    cv2.imshow("img1", img)

    img = val_dataset.get_filpbody_image(i)
    joints = val_dataset.get_filpbody_joints(i)
    joint_vis = val_dataset.get_filpbody_joint_vis(i)

    for jt, (j, jv) in enumerate(zip(joints, joint_vis)):
        if jv[0] > 0:
            cv2.putText(img, str(jt), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.circle(img,(int(j[0]), int(j[1])), 1, (0, 0, 255), 1)
    cv2.imshow("img2", img)
    cv2.waitKey(0)  
        