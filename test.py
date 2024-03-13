import torchvision.transforms as transforms
from dataset import *
import matplotlib.pyplot as plt
import cv2
import numpy as np

#model = HRNet(out_channels=21, base_channels=48)
#input = torch.randn(1, 3, 384, 288)
#output = model.forward(input)
#print(output.size())

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_annopath = "data\\annotations\\person_keypoints_wholebody_val.json"
val_imagepath = "data\\val2017"
val_dataset = COCOWholebody_BBox(val_annopath, val_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize]),
                                         image_height=288, image_width=384, heatmap_height=72, heatmap_width=96) 

for i in range(len(val_dataset)):
    img = val_dataset.get_preprocessed_image(i)
    targets = val_dataset.generate_heatmap_from_bbox(i)
    
    for cat, t in enumerate(targets):
        #plt.imshow(t)
        #plt.show()

        t *= 255
        t = np.array(t, dtype=np.uint8)
        resized_t = cv2.resize(t, (val_dataset.image_width, val_dataset.image_height))

        lower_thesh = 150
        upper_thesh = 255
        _, thresh = cv2.threshold(resized_t, lower_thesh, upper_thesh, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            (x, y, w, h) = cv2.boundingRect(c)

            w_s = int(w * 2.7)
            h_s = int(h * 2.7)
            x_s = int(x - (w_s - w) / 2)
            y_s = int(y - (h_s - h) / 2)

            (x, y, w, h) = (x_s, y_s, w_s, h_s)

            if cat == 0:
                #cv2.circle(img,(cx, cy), 8, (255, 0, 0), -1)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            if cat == 1:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            if cat == 2:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            if cat == 3:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 1)
        cv2.imshow("img", img)
        cv2.imshow("thresh", resized_t)
        cv2.waitKey(100) 
        