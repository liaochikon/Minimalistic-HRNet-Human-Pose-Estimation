import torchvision.transforms as transforms
from dataset import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn as nn

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

root_path = "data\\cmu_hand_manual"
val_dataset = CMU_Hand_Manual(root_path, transforms=transforms.Compose([transforms.ToTensor(), normalize])) 

print(len(val_dataset))
for image_transforms, targets, target_weights, misc in val_dataset:
    image_preprocess = misc['image_preprocess']
    #print(misc['joints'])
    #print(misc['joint_vis'])
    print(misc['flipped'])
    for jt, (j, jv) in enumerate(zip(misc['joints'], misc['joint_vis'])):
        if jv[0] > 0:
            cv2.putText(image_preprocess, str(jt), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.circle(image_preprocess,(int(j[0]), int(j[1])), 1, (0, 0, 255), 1)
        #cv2.putText(image_preprocess, str(jt), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        #cv2.circle(image_preprocess,(int(j[0]), int(j[1])), 1, (0, 0, 255), 1)   

    cv2.imshow("img1", image_preprocess)
    #plt.imshow(targets[0], cmap='hot', vmax=1, interpolation='nearest')
    #plt.show()


    #img = val_dataset.get_filpbody_image(i)
    #joints = val_dataset.get_filpbody_joints(i)
    #joint_vis = val_dataset.get_filpbody_joint_vis(i)
#
    #for jt, (j, jv) in enumerate(zip(joints, joint_vis)):
    #    if jv[0] > 0:
    #        cv2.putText(img, str(jt), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    #        cv2.circle(img,(int(j[0]), int(j[1])), 1, (0, 0, 255), 1)
    #cv2.imshow("img2", img)
    cv2.waitKey(0)  
