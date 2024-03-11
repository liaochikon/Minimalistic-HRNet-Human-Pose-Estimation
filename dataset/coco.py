import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

#annopath = "data\\annotations\\instances_val2017.json"
#imagepath = "data\\val2017"

#c = COCO(annopath)
#
#id = c.getCatIds('person')[0]
#cats = c.loadCats(1)
#imgids = c.getImgIds(catIds=1)
#imginfos = c.loadImgs(imgids[0])
#annids = c.getAnnIds(imgIds=imgids[0])
#anns = c.loadAnns(annids)
#
#print(len(c.imgs.keys()))

#img = cv2.imread(os.path.join(imagepath, imginfos[0]['file_name']))
#bbox = anns[0]['bbox']
#name = c.loadCats(anns[0]['category_id'])[0]['name']
#cv2.putText(img, name, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
#cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 1)
#cv2.imshow(imginfos[0]['file_name'], img)
#cv2.waitKey(0)

class COCOBasic(Dataset):
    def __init__(self, anno_path, image_root_path, transforms = None):
        self.anno_path = anno_path
        self.image_root_path = image_root_path
        self._transforms = transforms

        self._COCO = COCO(anno_path)

        self.image_ids = list(self._COCO.imgs.keys())

    def __getitem__(self, idx):
        anno_ids = self._COCO.getAnnIds(imgIds=self.image_ids[idx])
        image_path = self._COCO.loadImgs(self.image_ids[idx])[0]['file_name']
        image = cv2.imread(os.path.join(self.image_root_path, image_path))
        anno = self._COCO.loadAnns(anno_ids)
        return image, anno

    def __len__(self):
        return len(self.image_ids)
    
class COCOWholebody_BodyWithFeet(Dataset):
    def __init__(self, anno_path, image_root_path, 
                 num_joints = 23, 
                 image_height = 192, image_width = 256, 
                 heatmap_height = 48, heatmap_width = 64, heatmap_sigma = 1,
                 transforms = None):
        
        self.anno_path = anno_path
        self.image_root_path = image_root_path
        self.num_joints = num_joints
        self.image_height = image_height
        self.image_width = image_width
        self.heatmap_height = heatmap_height
        self.heatmap_width = heatmap_width
        self.heatmap_sigma = heatmap_sigma
        self._transforms = transforms

        self._COCO = COCO(anno_path)
        self.raw_image_ids = list(self._COCO.imgs.keys())

        self.image_ids = []
        self.image_paths = []
        self.bbox_list = []
        self.joints_list = []
        self.joint_vis_list = []
        for raw_image_id in self.raw_image_ids:
            anno_ids = self._COCO.getAnnIds(imgIds=raw_image_id)
            image_path = self._COCO.loadImgs(raw_image_id)[0]['file_name']
            image_path = os.path.join(self.image_root_path, image_path)
            people_anno = self._COCO.loadAnns(anno_ids)

            if len(people_anno) == 0:
                continue

            for person_anno in people_anno:
                if person_anno['num_keypoints'] == 0:
                    continue

                bbox = person_anno['bbox']
                body_keypoints = person_anno['keypoints']
                feet_keypoints = person_anno['foot_kpts']
                keypoints = np.array([*body_keypoints, *feet_keypoints], dtype=np.float)
                keypoints = keypoints.reshape((-1, 3))

                joints = keypoints.copy()
                joints[:, 2] = np.zeros(self.num_joints, dtype=np.float)

                joint_vis = np.zeros((self.num_joints, 3), dtype=np.float)
                for i, target_vis in enumerate(keypoints[:, 2]):
                    if target_vis > 1:
                        joint_vis[i][0] = 1.0
                        joint_vis[i][1] = 1.0
                        joint_vis[i][2] = 0.0
                    else:
                        joint_vis[i][0] = 0.0
                        joint_vis[i][1] = 0.0
                        joint_vis[i][2] = 0.0

                self.image_ids.append(raw_image_id)
                self.image_paths.append(image_path)
                self.bbox_list.append(bbox)
                self.joints_list.append(joints)
                self.joint_vis_list.append(joint_vis)

    def image_preprocess(self, image):
        scale = 0
        if image.shape[1] >= image.shape[0]:
            scale = self.image_width / image.shape[1]
        if image.shape[1] < image.shape[0]:
            scale = self.image_height / image.shape[0]

        p1 = np.float32([[0, 0],[image.shape[1], 0],[0, image.shape[0]]])
        p2 = np.float32([[0, 0],[int(image.shape[1] * scale), 0],[0, int(image.shape[0] * scale)]])
        M = cv2.getAffineTransform(p1, p2)

        warped_image = cv2.warpAffine(image, M, (self.image_width, self.image_height))

        return warped_image, M
    
    def generate_heatmap(self, joints, joint_vis, sigma = 2, use_different_joints_weight = False):
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

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.image_paths[idx]
        bbox = self.bbox_list[idx]
        joints = self.joints_list[idx]
        joint_vis = self.joint_vis_list[idx]

        image = cv2.imread(image_path)
        croped_image = image[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(bbox[0] + bbox[2])]
        image_preprocess, M = self.image_preprocess(croped_image)
        joints[:, 0] -= bbox[0]
        joints[:, 1] -= bbox[1]
        joints[:, :2] = np.matmul(M[:, :2], joints[:, :2].T).T

        targets, target_weights = self.generate_heatmap(joints, joint_vis, self.heatmap_sigma)

        if self._transforms:
            image_preprocess = self._transforms(image_preprocess)

        targets = torch.from_numpy(targets)
        target_weights = torch.from_numpy(target_weights)
        
        return image_preprocess, targets, target_weights, joints, joint_vis

    def __len__(self):
        return len(self.image_ids)

if __name__ == "__main__":
    annopath = "data\\annotations\\person_keypoints_wholebody_val.json"
    imagepath = "data\\val2017"
    c = COCOWholebody_BodyWithFeet(annopath, imagepath, image_height=256, image_width=192, heatmap_height=64, heatmap_width=48)

    for id in range(len(c)):
        img, targets, target_weights, _, _ = c[id]

        resized_img = cv2.resize(img, (c.heatmap_width, c.heatmap_height))
        resized_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        img_concat = resized_gray
        for t in targets:
            t *= 600
            t += resized_gray
            img_concat = np.hstack((img_concat, t))

        plt.imshow(img_concat, vmin=0, vmax=600)
        plt.show()

        #for i in range(len(targets)):
        #    t = targets[i]
        #    tw = target_weights[i]
        #    if int(tw[0]) == 0:
        #        cv2.circle(img, (int(t[0]), int(t[1])), 2, (0, 0, 255), -1)
        #    if int(tw[0]) == 1:
        #        cv2.circle(img, (int(t[0]), int(t[1])), 2, (0, 255, 0), -1)
        #    if int(tw[0]) == 2:
        #        cv2.circle(img, (int(t[0]), int(t[1])), 2, (255, 0, 0), -1)
        #cv2.imshow("s", img)
        #cv2.waitKey(0) 

    #for t in targets:
    #    bbox = t['bbox']
    #    category_id = t['category_id']
    #    cv2.putText(img, str(category_id), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    #    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 1)

    