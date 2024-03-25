import torch
import torchvision.transforms as transforms
from model import HRNet
from dataset import *
from util.joint_util import *
import cv2

#COCOWholebody model test config
########################################## Model test config start:
#batch_size = 1
#device = "cuda"
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
#val_annopath = "data\\coco_wholebody\\annotations\\coco_wholebody_val_v1.0.json"
#val_imagepath = "data\\coco_wholebody\\val2017"
#val_dataset = COCOWholebody_BodyWithFeet(val_annopath, val_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize]),
#                                         image_height=384, image_width=288, heatmap_height=96, heatmap_width=72)
#
#model = HRNet(base_channels=48, out_channels=23)
#model_dict = torch.load("weight/best_loss.pth")
#model.load_state_dict(model_dict['model_state_dict'])
########################################## Model test config end.

#HalpeFullbody model test config
######################################### Model test config start:
batch_size = 1
device = "cuda"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_annopath = "data\\halpe_fullbody\\annotations\\halpe_val_v1.json"
val_imagepath = "data\\halpe_fullbody\\val2017"
val_dataset = Halpe_Fullbody(val_annopath, val_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize]), image_height=384, image_width=288, heatmap_height=96, heatmap_width=72)

model = HRNet(base_channels=48, out_channels=26)
model_dict = torch.load("weight/best_acc.pth")
model.load_state_dict(model_dict['model_state_dict'])
######################################### Model test config end.

best_train_acc = 0.0
best_val_acc = 0.0

model.eval()
print_freq = 10
with torch.no_grad():
    for i, (input, targets, _, data_idx, flipped) in enumerate(val_dataset):
        targets = targets.reshape((1, targets.size(0),  targets.size(1), targets.size(2))).cuda(non_blocking=True)

        output = model(input.reshape((1, input.size(0),  input.size(1), input.size(2))))

        _, avg_acc, cnt, _ = accuracy(output.detach().cpu().numpy(),targets.detach().cpu().numpy())

        pred, maxvals = get_max_preds(output.detach().cpu().numpy())

        print("avg_acc : {}".format(avg_acc))

        image = []
        if flipped:
            image = val_dataset.get_filpbody_image(data_idx)
        else:
            image = val_dataset.get_preprocessed_image(data_idx)

        pred_ints = []
        for t, (p, v) in enumerate(zip(pred[0], maxvals[0])):
            pred_int = (int(p[0] / val_dataset.heatmap_width * val_dataset.image_width), int(p[1] / val_dataset.heatmap_height * val_dataset.image_height))
            cv2.putText(image, str(t), pred_int, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.circle(image, pred_int, 1, (0, 0, 255), 1)
            pred_ints.append(pred_int)
        cv2.line(image, pred_ints[0],  pred_ints[1],   (255, 255, 0), 2)
        cv2.line(image, pred_ints[0],  pred_ints[2],   (255, 255, 0), 2)
        cv2.line(image, pred_ints[1],  pred_ints[3],   (255, 255, 0), 2)
        cv2.line(image, pred_ints[2],  pred_ints[4],   (255, 255, 0), 2)
        cv2.line(image, pred_ints[5],  pred_ints[6],   (255, 255, 0), 2)
        cv2.line(image, pred_ints[11], pred_ints[12], (255, 255, 0), 2)
        cv2.line(image, pred_ints[6],  pred_ints[8],   (255, 255, 0), 2)
        cv2.line(image, pred_ints[8],  pred_ints[10],  (255, 255, 0), 2)
        cv2.line(image, pred_ints[5],  pred_ints[7],   (255, 255, 0), 2)
        cv2.line(image, pred_ints[7],  pred_ints[9],   (255, 255, 0), 2)
        cv2.line(image, pred_ints[12], pred_ints[14], (255, 255, 0), 2)
        cv2.line(image, pred_ints[14], pred_ints[16], (255, 255, 0), 2)
        cv2.line(image, pred_ints[11], pred_ints[13], (255, 255, 0), 2)
        cv2.line(image, pred_ints[13], pred_ints[15], (255, 255, 0), 2)
        cv2.line(image, pred_ints[0],  pred_ints[17], (255, 255, 0), 2)
        cv2.line(image, pred_ints[0],  pred_ints[18], (255, 255, 0), 2)
        cv2.line(image, pred_ints[18], pred_ints[19], (255, 255, 0), 2)
        cv2.line(image, pred_ints[15], pred_ints[24], (255, 255, 0), 2)
        cv2.line(image, pred_ints[15], pred_ints[22], (255, 255, 0), 2)
        cv2.line(image, pred_ints[15], pred_ints[20], (255, 255, 0), 2)
        cv2.line(image, pred_ints[16], pred_ints[25], (255, 255, 0), 2)
        cv2.line(image, pred_ints[16], pred_ints[23], (255, 255, 0), 2)
        cv2.line(image, pred_ints[16], pred_ints[21], (255, 255, 0), 2)
        cv2.imshow("img", image)
        cv2.waitKey(0) 