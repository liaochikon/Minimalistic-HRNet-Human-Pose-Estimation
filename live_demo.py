import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2

from model import HRNet
from util.joint_util import *
from faster_rcnn import predict as faster_rcnn_predict
from faster_rcnn import person_boxes

def get_affine(bbox, image_width, image_height):
    p1 = np.float32([[int(bbox[0]), int(bbox[1])],[int(bbox[0] + bbox[2]), int(bbox[1])],[int(bbox[0]), int(bbox[1] + bbox[3])]])
    p2 = np.float32([[0, 0],[image_width, 0],[0, image_height]])
    M = cv2.getAffineTransform(p1, p2)
    return M

#HalpeFullbody model config
######################################### Live demo config start:
cap = cv2.VideoCapture(0) #You can change webcam or read prerecorded video in the line.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

image_height=384
image_width=288
heatmap_height=96
heatmap_width=72

device = "cuda"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
T = transforms.Compose([transforms.ToTensor(), normalize])

model = HRNet(base_channels=48, out_channels=26)
model_dict = torch.load("weight/best_acc.pth")
######################################### Live demo config end.

model.load_state_dict(model_dict['model_state_dict'])
model = torch.nn.DataParallel(model, device_ids=(0,)).cuda()
model.eval()

faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=800)
faster_rcnn.eval().to(device)

while True:
    ret, frame = cap.read()
    boxes, classes, labels = faster_rcnn_predict(frame, faster_rcnn, device, 0.8)
    frame, p_boxes = person_boxes(boxes, labels, frame)

    results = {'preds' : [], 'maxvals' : []}
    for p, p_box in enumerate(p_boxes):
        w = p_box[2] - p_box[0]
        h = p_box[3] - p_box[1]
        scale = max([w / image_width, h / image_height]) * 1.2
        center = (p_box[0] + w / 2, p_box[1] + h / 2)
        clean_bbox_topleft = (center[0] - image_width / 2 * scale, center[1] - image_height / 2 * scale)
        clean_bbox = [clean_bbox_topleft[0], clean_bbox_topleft[1], image_width * scale, image_height * scale]

        M = get_affine(clean_bbox, image_width, image_height)
        image_preprocess = cv2.warpAffine(frame, M, (image_width, image_height))
        image_transformed = T(image_preprocess)

        outputs = model(image_transformed.reshape((1, image_transformed.size(0),  image_transformed.size(1), image_transformed.size(2))))

        preds, maxvals = get_max_preds(outputs.detach().cpu().numpy())
        preds = preds[0]
        maxvals = maxvals[0]
        preds[:, 0] *= image_width / heatmap_width
        preds[:, 1] *= image_height / heatmap_height
        preds = np.matmul(np.linalg.inv(M[:, :2]), preds.T).T
        preds[:, 0] += clean_bbox_topleft[0]
        preds[:, 1] += clean_bbox_topleft[1]
        results['preds'].append(preds)
        results['maxvals'].append(maxvals)

    for person_num, (pred, maxval) in enumerate(zip(results['preds'], results['maxvals'])):
        for i, (p, v) in enumerate(zip(pred, maxval)):
            #if v[0] < 0.3:
            #    continue
            cv2.putText(frame, str(i), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.circle(frame,(int(p[0]), int(p[1])), 1, (0, 0, 255), 2)
        pred_int = pred.astype(int)
        cv2.line(frame, pred_int[0], pred_int[1],   (255, 255, 0), 2)
        cv2.line(frame, pred_int[0], pred_int[2],   (255, 255, 0), 2)
        cv2.line(frame, pred_int[1], pred_int[3],   (255, 255, 0), 2)
        cv2.line(frame, pred_int[2], pred_int[4],   (255, 255, 0), 2)
        cv2.line(frame, pred_int[5], pred_int[6],   (255, 255, 0), 2)
        cv2.line(frame, pred_int[11], pred_int[12], (255, 255, 0), 2)
        cv2.line(frame, pred_int[6], pred_int[8],   (255, 255, 0), 2)
        cv2.line(frame, pred_int[8], pred_int[10],  (255, 255, 0), 2)
        cv2.line(frame, pred_int[5], pred_int[7],   (255, 255, 0), 2)
        cv2.line(frame, pred_int[7], pred_int[9],   (255, 255, 0), 2)
        cv2.line(frame, pred_int[12], pred_int[14], (255, 255, 0), 2)
        cv2.line(frame, pred_int[14], pred_int[16], (255, 255, 0), 2)
        cv2.line(frame, pred_int[11], pred_int[13], (255, 255, 0), 2)
        cv2.line(frame, pred_int[13], pred_int[15], (255, 255, 0), 2)

        cv2.line(frame, pred_int[0], pred_int[17], (255, 255, 0), 2)
        cv2.line(frame, pred_int[0], pred_int[18], (255, 255, 0), 2)
        cv2.line(frame, pred_int[18], pred_int[19], (255, 255, 0), 2)

        cv2.line(frame, pred_int[15], pred_int[24], (255, 255, 0), 2)
        cv2.line(frame, pred_int[15], pred_int[22], (255, 255, 0), 2)
        cv2.line(frame, pred_int[15], pred_int[20], (255, 255, 0), 2)

        cv2.line(frame, pred_int[16], pred_int[25], (255, 255, 0), 2)
        cv2.line(frame, pred_int[16], pred_int[23], (255, 255, 0), 2)
        cv2.line(frame, pred_int[16], pred_int[21], (255, 255, 0), 2)
        
        
    cv2.imshow("img", frame)
    cv2.waitKey(1) 