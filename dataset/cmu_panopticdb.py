import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import json
import random

class CMU_Panopticdb(Dataset):
    def __init__(self, root_path,
                 num_joints = 21,
                 json_name = "hands_v143_14817.json",
                 image_height = 288, image_width = 288,
                 heatmap_height = 72, heatmap_width = 72, heatmap_sigma = 2,
                 flip_prob = 0.5,
                 transforms = None):
        
        self.root_path = root_path
        self.num_joints = num_joints
        self.json_name = json_name
        self.image_height = image_height
        self.image_width = image_width
        self.heatmap_height = heatmap_height
        self.heatmap_width = heatmap_width
        self.heatmap_sigma = heatmap_sigma
        self.flip_prob = flip_prob
        self._transforms = transforms
        
        self.image_paths = []
        self.image_affines = []
        self.bbox_list = []
        self.clean_bbox_list = []
        self.joints_list = []
        self.joint_vis_list = []
        
        json_dict = {}
        json_path = os.path.join(self.root_path, self.json_name)
        with open(json_path, "r") as readfile:
            json_dict = json.load(readfile)

        for anno in json_dict['root']:
            image_path = os.path.join(self.root_path, anno['img_paths'])
            if os.path.isfile(image_path) == False:
                continue
            if anno['isValidation']:
                print(anno['isValidation'])

            hand_keypoints = np.array(anno['joint_self'], dtype=np.float)
            joints = hand_keypoints.copy()
            joints[:, 2] = np.zeros(self.num_joints, dtype=np.float)
            joint_vis = np.zeros((self.num_joints, 3), dtype=np.float)

            valid_hand_keypoints = []
            for i, target_vis in enumerate(hand_keypoints[:, 2]):
                if target_vis > 0:
                    joint_vis[i][0] = 1.0
                    joint_vis[i][1] = 1.0
                    joint_vis[i][2] = 0.0
                    valid_hand_keypoints.append(hand_keypoints[i])
                else:
                    joint_vis[i][0] = 0.0
                    joint_vis[i][1] = 0.0
                    joint_vis[i][2] = 0.0
            valid_hand_keypoints = np.array(valid_hand_keypoints)
            
            x1 = min(valid_hand_keypoints[:, 0])
            y1 = min(valid_hand_keypoints[:, 1])
            x2 = max(valid_hand_keypoints[:, 0])
            y2 = max(valid_hand_keypoints[:, 1])
            bbox = (x1, y1, x2 - x1, y2 - y1)
            scale = max([bbox[2] / self.image_width, bbox[3] / self.image_height]) + 0.2
            center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            clean_bbox_topleft = (center[0] - self.image_width / 2 * scale, center[1] - self.image_height / 2 * scale)
            clean_bbox = [clean_bbox_topleft[0], clean_bbox_topleft[1], self.image_width * scale, self.image_height * scale]
            image_affine = self.get_affine(clean_bbox)

            self.image_paths.append(image_path)
            self.image_affines.append(image_affine)
            self.bbox_list.append(bbox)
            self.clean_bbox_list.append(clean_bbox)
            self.joints_list.append(joints)
            self.joint_vis_list.append(joint_vis)

    def get_affine(self, bbox):
        p1 = np.float32([[int(bbox[0]), int(bbox[1])],[int(bbox[0] + bbox[2]), int(bbox[1])],[int(bbox[0]), int(bbox[1] + bbox[3])]])
        p2 = np.float32([[0, 0],[self.image_width, 0],[0, self.image_height]])
        M = cv2.getAffineTransform(p1, p2)
        return M
    
    def generate_heatmap_from_joints(self, joints, joint_vis, sigma = 2, use_different_joints_weight = False):
        target_weights = np.ones((self.num_joints, 1), dtype=np.float)
        target_weights[:, 0] = joint_vis[:, 0]

        targets = np.zeros((self.num_joints,
                            self.heatmap_height,
                            self.heatmap_width),
                            dtype=np.float)

        tmp_size = sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride_x = self.image_width / self.heatmap_width
            feat_stride_y = self.image_height / self.heatmap_height
            mu_x = int(joints[joint_id][0] / feat_stride_x + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride_y + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_width or ul[1] >= self.heatmap_height \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weights[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_width) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_height) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_width)
            img_y = max(0, ul[1]), min(br[1], self.heatmap_height)

            v = target_weights[joint_id]
            if v > 0.5:
                targets[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        if use_different_joints_weight:
            target_weights = np.multiply(target_weights, self.joints_weight)
        return targets, target_weights
    
    def get_filpbody_image(self, idx):
        image_path = self.image_paths[idx]
        image_affine = self.image_affines[idx].copy()
        image = cv2.imread(image_path)
        warped_image = cv2.warpAffine(image, image_affine, (self.image_width, self.image_height))
        fliped_image = cv2.flip(warped_image, 1)
        return fliped_image

    def get_filpbody_joints(self, idx):
        image_affine = self.image_affines[idx].copy()
        clean_bbox = self.clean_bbox_list[idx].copy()
        joints = self.joints_list[idx].copy()

        joints[:, 0] -= clean_bbox[0]
        joints[:, 1] -= clean_bbox[1]
        joints[:, :2] = np.matmul(image_affine[:, :2], joints[:, :2].T).T

        joints[:, 0] *= -1 
        joints[:, 0] += self.image_width
        return joints
    
    def get_filpbody_joint_vis(self, idx):
        joint_vis = self.joint_vis_list[idx].copy()
        return joint_vis

    def get_preprocessed_image(self, idx):
        image_path = self.image_paths[idx]
        image_affine = self.image_affines[idx].copy()
        image = cv2.imread(image_path)
        warped_image = cv2.warpAffine(image, image_affine, (self.image_width, self.image_height))
        return warped_image
    
    def get_preprocessed_joints(self, idx):
        image_affine = self.image_affines[idx].copy()
        clean_bbox = self.clean_bbox_list[idx].copy()
        joints = self.joints_list[idx].copy()

        joints[:, 0] -= clean_bbox[0]
        joints[:, 1] -= clean_bbox[1]
        joints[:, :2] = np.matmul(image_affine[:, :2], joints[:, :2].T).T
        return joints
    
    def get_preprocessed_joint_vis(self, idx):
        joint_vis = self.joint_vis_list[idx].copy()
        return joint_vis

    def __getitem__(self, idx):
        image_preprocess = []
        joints = []
        joint_vis = []
        flipped = None
        if random.random() <= self.flip_prob:
            image_preprocess = self.get_filpbody_image(idx)
            joints = self.get_filpbody_joints(idx)
            joint_vis = self.get_filpbody_joint_vis(idx)
            flipped = True
        else:
            image_preprocess = self.get_preprocessed_image(idx)
            joints = self.get_preprocessed_joints(idx)
            joint_vis = self.get_preprocessed_joint_vis(idx)
            flipped = False

        targets, target_weights = self.generate_heatmap_from_joints(joints, joint_vis, self.heatmap_sigma)

        if self._transforms:
            image_transforms = self._transforms(image_preprocess)
        targets = torch.from_numpy(targets)
        target_weights = torch.from_numpy(target_weights)

        misc = {'image_preprocess' : image_preprocess,
                'idx' : idx,
                'flipped' : flipped,
                'joints' : joints,
                'joint_vis' : joint_vis}
        
        return image_transforms, targets, target_weights, misc

    def __len__(self):
        return len(self.image_paths)