import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
import cv2
import os

import matplotlib.pyplot as plt

class COCOWholebody_BBox(Dataset):
    def __init__(self, anno_path, image_root_path,
                 num_joints = 21,
                 image_height = 192, image_width = 256,
                 heatmap_height = 48, heatmap_width = 64, heatmap_kernel_size = (5, 5), heatmap_sigma = 2,
                 transforms = None):
        
        self.anno_path = anno_path
        self.image_root_path = image_root_path
        self.num_joints = num_joints
        self.image_height = image_height
        self.image_width = image_width
        self.heatmap_height = heatmap_height
        self.heatmap_width = heatmap_width
        self.heatmap_kernel_size = heatmap_kernel_size
        self.heatmap_sigma = heatmap_sigma
        self._transforms = transforms

        self._COCO = COCO(anno_path)
        self.raw_image_ids = list(self._COCO.imgs.keys())

        self.image_ids = []
        self.image_sizes = []
        self.image_paths = []
        self.image_affines = []
        self.person_bbox_list = []
        self.face_bbox_list = []
        self.lefthand_bbox_list = []
        self.righthand_bbox_list = []
        for raw_image_id in self.raw_image_ids:
            anno_ids = self._COCO.getAnnIds(imgIds=raw_image_id)
            image_info = self._COCO.loadImgs(raw_image_id)[0]
            image_sizes = (image_info['height'], image_info['width'])
            image_path = image_info['file_name']
            image_path = os.path.join(self.image_root_path, image_path)
            image_affine = self.get_affine(image_sizes)
            people_anno = self._COCO.loadAnns(anno_ids)

            if len(people_anno) == 0:
                continue

            person_bbox_list = []
            face_bbox_list = []
            lefthand_bbox_list = []
            righthand_bbox_list = []
            for person_anno in people_anno:
                person_bbox_list.append(person_anno['bbox'])
                face_bbox_list.append(person_anno['face_box'])
                lefthand_bbox_list.append(person_anno['lefthand_box'])
                righthand_bbox_list.append(person_anno['righthand_box'])
            
            self.image_ids.append(raw_image_id)
            self.image_sizes.append(image_sizes)
            self.image_paths.append(image_path)
            self.image_affines.append(image_affine)
            self.person_bbox_list.append(person_bbox_list)
            self.face_bbox_list.append(face_bbox_list)
            self.lefthand_bbox_list.append(lefthand_bbox_list)
            self.righthand_bbox_list.append(righthand_bbox_list)

    def get_affine(self, image_size):
        center = (image_size[1] / 2, image_size[0] / 2)
        scale = max([image_size[1] / self.image_width, image_size[0] / self.image_height])
        bbox_topleft = (center[0] - self.image_width / 2 * scale, center[1] - self.image_height / 2 * scale)
        bbox = [bbox_topleft[0], bbox_topleft[1], self.image_width * scale, self.image_height * scale]

        p1 = np.float32([[int(bbox[0]), int(bbox[1])],[int(bbox[0] + bbox[2]), int(bbox[1])],[int(bbox[0]), int(bbox[1] + bbox[3])]])
        p2 = np.float32([[0, 0],[self.image_width, 0],[0, self.image_height]])
        M = cv2.getAffineTransform(p1, p2)
        return M
    
    def get_bbox_feature_map(self, bbox_int, sigma_x_ratio = 0.5, sigma_y_ratio = 0.5):
        def gaus2d(x = 0, y = 0, mx = bbox_int[2] / 2, my = bbox_int[3] / 2, sx = bbox_int[2] * sigma_x_ratio, sy = bbox_int[3] * sigma_y_ratio):
            return 1 / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

        x = np.linspace(0, bbox_int[2], num=bbox_int[2])
        y = np.linspace(0, bbox_int[3], num=bbox_int[3])

        x, y = np.meshgrid(x, y)
        feature_map = gaus2d(x, y)
        feature_map /= np.max(feature_map)

        return feature_map
    
    def bbox_preprocess(self, bbox, image_size):
        if bbox[2] <= 0 or bbox[3] <= 0:
            return []
            
        bbox_int = np.array(bbox, dtype=np.int)

        if bbox_int[0] < 0:
            bbox_int[0] = 0
            bbox_int[2] += bbox_int[0]
        if bbox_int[1] < 0:
            bbox_int[1] = 0
            bbox_int[3] += bbox_int[1]

        if bbox_int[2] + bbox_int[0] > image_size[1]:
            bbox_int[2] = image_size[1] - bbox_int[0]

        if bbox_int[3] + bbox_int[1] > image_size[0]:
            bbox_int[3] = image_size[0] - bbox_int[1]
        
        return bbox_int

    def generate_heatmap_from_bbox(self, idx):
        image_size = self.image_sizes[idx]
        image_affine = self.image_affines[idx].copy()
        person_bbox_list = self.person_bbox_list[idx].copy()
        face_bbox_list = self.face_bbox_list[idx].copy()
        lefthand_bbox_list = self.lefthand_bbox_list[idx].copy()
        righthand_bbox_list = self.righthand_bbox_list[idx].copy()

        targets = np.zeros((4, image_size[0], image_size[1]), dtype=np.float)
        #person bbox:
        for person_bbox in person_bbox_list:
            person_bbox_int = self.bbox_preprocess(person_bbox, image_size)
            if len(person_bbox_int) == 0:
                continue

            feature_map = self.get_bbox_feature_map(person_bbox_int, sigma_x_ratio = 0.2, sigma_y_ratio = 0.2)
            targets[0,
                    person_bbox_int[1]:person_bbox_int[1] + person_bbox_int[3], 
                    person_bbox_int[0]:person_bbox_int[0] + person_bbox_int[2]] = feature_map
        
        #face bbox:
        for face_bbox in face_bbox_list:
            face_bbox_int = self.bbox_preprocess(face_bbox, image_size)
            if len(face_bbox_int) == 0:
                continue
                
            feature_map = self.get_bbox_feature_map(face_bbox_int, sigma_x_ratio = 0.3, sigma_y_ratio = 0.3)
            targets[1, 
                    face_bbox_int[1]:face_bbox_int[1] + face_bbox_int[3], 
                    face_bbox_int[0]:face_bbox_int[0] + face_bbox_int[2]] = feature_map
        
        #lefthand bbox:
        for lefthand_bbox in lefthand_bbox_list:
            lefthand_bbox_int = self.bbox_preprocess(lefthand_bbox, image_size)
            if len(lefthand_bbox_int) == 0:
                continue

            feature_map = self.get_bbox_feature_map(lefthand_bbox_int, sigma_x_ratio = 0.3, sigma_y_ratio = 0.3)
            targets[2,
                    lefthand_bbox_int[1]:lefthand_bbox_int[1] + lefthand_bbox_int[3], 
                    lefthand_bbox_int[0]:lefthand_bbox_int[0] + lefthand_bbox_int[2]] = feature_map
        
        #righthand bbox:
        for righthand_bbox in righthand_bbox_list:
            righthand_bbox_int = self.bbox_preprocess(righthand_bbox, image_size)
            if len(righthand_bbox_int) == 0:
                continue

            feature_map = self.get_bbox_feature_map(righthand_bbox_int, sigma_x_ratio = 0.3, sigma_y_ratio = 0.3)
            targets[3,
                    righthand_bbox_int[1]:righthand_bbox_int[1] + righthand_bbox_int[3], 
                    righthand_bbox_int[0]:righthand_bbox_int[0] + righthand_bbox_int[2]] = feature_map
        
        targets_preprocessed = []
        for cat in range(4):
            target_warped = cv2.warpAffine(targets[cat], image_affine, (self.image_width, self.image_height))
            target_resized = cv2.resize(target_warped, (self.heatmap_width, self.heatmap_height))
            
            target_max = np.max(target_resized)
            if target_max > 0:
                target_resized = target_resized / target_max
            targets_preprocessed.append(target_resized)
        
        return targets_preprocessed
            
    
    def get_preprocessed_image(self, idx):
        image_path = self.image_paths[idx]
        image_affine = self.image_affines[idx]
        image = cv2.imread(image_path)
        image_preprocess = cv2.warpAffine(image, image_affine, (self.image_width, self.image_height))

        return image_preprocess
    
    def __getitem__(self, idx):
        image_preprocess = self.get_preprocessed_image(idx)
        targets_preprocessed = self.generate_heatmap_from_bbox(idx)

        if self._transforms:
            image_preprocess = self._transforms(image_preprocess)

        targets = torch.from_numpy(targets_preprocessed)
        
        return image_preprocess, targets, idx

    def __len__(self):
        return len(self.image_ids)




    