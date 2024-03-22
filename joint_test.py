import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import HRNet
from dataset import COCOWholebody_BodyWithFeetAndPalm
from util.joint_util import *
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_output_result(dataset, outputs, targets, data_idx, image_name, epoch, batch_num):
    img_output = []
    for i, (output, target, data_id) in enumerate(zip(outputs, targets, data_idx)):
        img = dataset.get_preprocessed_image(data_id)
        resized_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 2
        #img_concat = resized_gray
        target_concat = resized_gray
        output_concat = resized_gray
        for t, ta in zip(output, target):
            t = t.detach().cpu().numpy()
            ta = ta.detach().cpu().numpy()
            t = cv2.resize(t, (img.shape[1], img.shape[0]))
            ta = cv2.resize(ta, (img.shape[1], img.shape[0]))
            t *= 255 / 2
            ta *= 255 / 2
            t += resized_gray
            ta += resized_gray
            #img_concat = np.hstack((img_concat, resized_gray))
            target_concat = np.hstack((target_concat, ta))
            output_concat = np.hstack((output_concat, t))
        if i == 0:
            img_output = np.vstack((target_concat, output_concat))
        else:
            img_output = np.vstack((img_output, target_concat))
            img_output = np.vstack((img_output, output_concat)) 
    plt.imshow(img_output, vmin=0, vmax=255)
    plt.savefig('log/' + image_name + '_' + str(epoch) + '_' + str(batch_num) + '.png', dpi=500)

batch_size = 1
device = "cuda"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_annopath = "data\\coco_wholebody\\annotations\\coco_wholebody_val_v1.0.json"
val_imagepath = "data\\coco_wholebody\\val2017"
val_dataset = COCOWholebody_BodyWithFeetAndPalm(val_annopath, val_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize]),
                                         image_height=384, image_width=288, heatmap_height=96, heatmap_width=72)

model = HRNet(base_channels=48, out_channels=27)
model_dict = torch.load("weight/best_acc.pth")
model.load_state_dict(model_dict['model_state_dict'])

best_train_acc = 0.0
best_val_acc = 0.0

model.eval()
print_freq = 10
with torch.no_grad():
    for i, (input, targets, _, data_idx, flipped) in enumerate(val_dataset):
        targets = targets.reshape((1, targets.size(0),  targets.size(1), targets.size(2))).cuda(non_blocking=True)

        output = model(input.reshape((1, input.size(0),  input.size(1), input.size(2))))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),targets.detach().cpu().numpy())

        pred_list = []
        cat_list = []

        #for j, t in enumerate(output[0]):
        #    t = t.detach().cpu().numpy()
        #    
        #    t = cv2.resize(t, (val_dataset.image_width, val_dataset.image_height))
        #    t *= 255
        #    t = np.floor(t)
        #    t[np.where(t < 0)] = 0
        #    t = np.array(t, dtype=np.uint8)
        #    #plt.imshow(t)
        #    #plt.show()
        #    cv2.imshow("img", t)
        #    cv2.waitKey(0) 
        image = []
        if flipped:
            image = val_dataset.get_filpbody_image(data_idx)
        else:
            image = val_dataset.get_preprocessed_image(data_idx)
        
        for t, p in enumerate(pred[0]):
            print(p)
            cv2.putText(image, str(t), (int(p[0] / val_dataset.heatmap_width * val_dataset.image_width), int(p[1] / val_dataset.heatmap_height * val_dataset.image_height)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

            cv2.circle(image,(int(p[0] / val_dataset.heatmap_width * val_dataset.image_width), int(p[1] / val_dataset.heatmap_height * val_dataset.image_height)), 1, (0, 0, 255), 1)

            #cv2.rectangle(image, (20, 60), (120, 160), (0, 255, 0), 2)

        cv2.imshow("img", image)
        cv2.waitKey(0) 

        #if (i + 1) % print_freq == 0:
        #    get_output_result(val_dataset, outputs, targets, data_idx, "val", 0, i + 1)