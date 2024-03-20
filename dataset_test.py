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

val_annopath = "data\\annotations\\person_keypoints_wholebody_val.json"
val_imagepath = "data\\val2017"
val_dataset = COCOWholebody_BodyWithFeetAndPalm(val_annopath, val_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize])) 

for i in range(len(val_dataset)):
    img = val_dataset.get_preprocessed_image(i)
    joints = val_dataset.get_preprocessed_joints(i)

    #img = val_dataset.get_filpbody_image(i)
    #joints = val_dataset.get_filpbody_joints(i)

    for jt, j in enumerate(joints):
        cv2.putText(img, str(jt), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.circle(img,(int(j[0]), int(j[1])), 1, (0, 0, 255), 1)
    cv2.imshow("img", img)
    cv2.waitKey(0)  
        