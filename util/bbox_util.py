import numpy as np
import cv2

def accuracy(outputs, targets, image_width, image_height):
    pred_list = []
    acc_list = []
    for output, target in zip(outputs, targets):
        for cat, (o, t) in enumerate(zip(output, target)):
            labels_t, bboxs_t = target_to_bboxs(t, cat, image_width, image_height)
            bboxs_t = xywh_to_xyxy(bboxs_t)
            labels_o, bboxs_o = target_to_bboxs(o, cat, image_width, image_height)
            bboxs_o = xywh_to_xyxy(bboxs_o)

            bboxs_o = sort_best_bboxs(bboxs_t, bboxs_o)
            acc = bboxs_accuracy(bboxs_t, bboxs_o)

            if np.isnan(acc):
                acc = 0

            pred_list.append(bboxs_o)
            acc_list.append(acc)

    avg_acc = np.mean(acc_list)

    return pred_list, acc_list, avg_acc


def bboxs_accuracy(bboxs_t, bboxs_o):
    iou_list = []
    for bbox_t, bbox_o in zip(bboxs_t, bboxs_o):
        iou = bb_intersection_over_union(bbox_t, bbox_o)
        
        iou_list.append(iou)
    
    acc = np.mean(iou_list)
    return acc

def bb_intersection_over_union(boxa, boxb):
        xa = max(boxa[0], boxb[0])
        ya = max(boxa[1], boxb[1])
        xb = min(boxa[2], boxb[2])
        yb = min(boxa[3], boxb[3])

        inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)
        boxa_area = (boxa[2] - boxa[0] + 1) * (boxa[3] - boxa[1] + 1)
        boxb_area = (boxb[2] - boxb[0] + 1) * (boxb[3] - boxb[1] + 1)
        iou = inter_area / float(boxa_area + boxb_area - inter_area)

        return iou

def target_to_bboxs(target, cat, image_width, image_height, lower_thesh = 150, upper_thesh = 255, bbox_scale_ratio = 2.7):
    labels = []
    bboxs = []

    target *= 255
    target = np.array(target, dtype=np.uint8)
    resized_t = cv2.resize(target, (image_width, image_height))

    _, thresh = cv2.threshold(resized_t, lower_thesh, upper_thesh, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        #cx = int(M["m10"] / M["m00"])
        #cy = int(M["m01"] / M["m00"])
        (x, y, w, h) = cv2.boundingRect(c)

        w_s = int(w * bbox_scale_ratio)
        h_s = int(h * bbox_scale_ratio)
        x_s = int(x - (w_s - w) / 2)
        y_s = int(y - (h_s - h) / 2)

        labels.append(cat)
        bboxs.append((x_s, y_s, w_s, h_s))

    return np.array(labels, dtype=np.float), np.array(bboxs, dtype=np.float)

def xywh_to_xyxy(xywh_bboxs):
    xyxy_bboxs = []
    for bbox in xywh_bboxs:
        (x, y, w, h) = bbox
        x_s = x + w
        y_s = y + h
        xyxy_bboxs.append((x, y, x_s, y_s))

    return np.array(xyxy_bboxs, dtype=np.float)

def sort_best_bboxs(bboxsa, bboxsb):
    best_bboxs = []
    for bboxa in bboxsa:
        best_iou = 0.0
        best_bbox = (0, 0, 0, 0)
        for bboxb in bboxsb:
            iou = bb_intersection_over_union(bboxa, bboxb)
            if iou > best_iou:
                best_iou = iou
                best_bbox = bboxb
        best_bboxs.append(best_bbox)

    return np.array(best_bboxs, dtype=np.float)
    
        