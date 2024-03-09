from torch.utils.data import Dataset
from pycocotools.coco import COCO
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
        image = cv2.imread(os.path.join(imagepath, image_path))
        anno = self._COCO.loadAnns(anno_ids)
        return image, anno

    def __len__(self):
        return len(self.image_ids)
    
class COCOWholebody_BodyWithFeet(Dataset):
    def __init__(self, anno_path, image_root_path, image_height = 192, image_width = 256, transforms = None):
        self.anno_path = anno_path
        self.image_root_path = image_root_path
        self.image_height = image_height
        self.image_width = image_width
        self._transforms = transforms

        self._COCO = COCO(anno_path)

        self.image_ids = list(self._COCO.imgs.keys())

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

        if self._transforms:
            warped_image = self._transforms(warped_image)

        return warped_image, M

    def anno_preprocess(self, people_anno, M):
        #No keypoint in this anno
        if len(people_anno) == 0:
            return False, [np.zeros((23, 3), dtype=np.float)], [np.zeros((23, 3), dtype=np.float)]
        
        targets = []
        target_weights = []
        for person_anno in people_anno:
            if person_anno['foot_valid'] == False:
                continue
            body_keypoints = person_anno['keypoints']
            feet_keypoints = person_anno['foot_kpts']
            keypoints = np.array([*body_keypoints, *feet_keypoints], dtype=np.float)
            keypoints = keypoints.reshape((-1, 3))
            keypoints[:, :2] = np.matmul(M[:, :2], keypoints[:, :2].T).T

            target = keypoints.copy()
            target[:, 2] = np.zeros(23, dtype=np.float)

            target_weight = np.zeros((23, 3), dtype=np.float)
            for i, target_vis in enumerate(keypoints[:, 2]):
                if target_vis > 1:
                    target_weight[i][0] = 1.0
                    target_weight[i][1] = 1.0
                    target_weight[i][2] = 0.0
                else:
                    target_weight[i][0] = 0.0
                    target_weight[i][1] = 0.0
                    target_weight[i][2] = 0.0

            targets.append(target)
            target_weights.append(target_weight)

        if len(targets) == 0:
            return False, [np.zeros((23, 3), dtype=np.float)], [np.zeros((23, 3), dtype=np.float)]

        return True, targets, target_weights

    def __getitem__(self, idx):
        anno_ids = self._COCO.getAnnIds(imgIds=self.image_ids[idx])
        image_path = self._COCO.loadImgs(self.image_ids[idx])[0]['file_name']
        image = cv2.imread(os.path.join(imagepath, image_path))
        anno = self._COCO.loadAnns(anno_ids)

        image_preprocess, M = self.image_preprocess(image)
        ret, targets, target_weights = self.anno_preprocess(anno, M)

        return image_preprocess, targets, target_weights

    def __len__(self):
        return len(self.image_ids)

if __name__ == "__main__":
    annopath = "data\\annotations\\person_keypoints_wholebody_val.json"
    imagepath = "data\\val2017"
    c = COCOWholebody_BodyWithFeet(annopath, imagepath)


    for id in range(len(c)):
        img, targets, target_weights = c[id]
        
        for i in range(len(targets)):
            for j in range(len(targets[i])):
                t = targets[i][j]
                tw = target_weights[i][j]
                if int(tw[0]) == 0:
                    cv2.circle(img, (int(t[0]), int(t[1])), 2, (0, 0, 255), -1)
                if int(tw[0]) == 1:
                    cv2.circle(img, (int(t[0]), int(t[1])), 2, (0, 255, 0), -1)
                if int(tw[0]) == 2:
                    cv2.circle(img, (int(t[0]), int(t[1])), 2, (255, 0, 0), -1)
        cv2.imshow("s", img)
        cv2.waitKey(0) 
    #for t in targets:
    #    bbox = t['bbox']
    #    category_id = t['category_id']
    #    cv2.putText(img, str(category_id), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    #    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 1)

    