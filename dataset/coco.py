from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import os

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

#if __name__ == "__main__":
#    annopath = "data\\annotations\\person_keypoints_wholebody_val.json"
#    imagepath = "data\\val2017"
#    c = COCOWholebody_BodyWithFeet(annopath, imagepath, image_height=256, image_width=192, heatmap_height=64, heatmap_width=48)
#
#    for id in range(len(c)):
#        img, targets, target_weights, _, _ = c[id]
#
#        resized_img = cv2.resize(img, (c.heatmap_width, c.heatmap_height))
#        resized_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
#
#        img_concat = resized_gray
#        for t in targets:
#            t *= 600
#            t += resized_gray
#            img_concat = np.hstack((img_concat, t))
#
#        plt.imshow(img_concat, vmin=0, vmax=600)
#        plt.show()
#
#        #for i in range(len(targets)):
#        #    t = targets[i]
#        #    tw = target_weights[i]
#        #    if int(tw[0]) == 0:
#        #        cv2.circle(img, (int(t[0]), int(t[1])), 2, (0, 0, 255), -1)
#        #    if int(tw[0]) == 1:
#        #        cv2.circle(img, (int(t[0]), int(t[1])), 2, (0, 255, 0), -1)
#        #    if int(tw[0]) == 2:
#        #        cv2.circle(img, (int(t[0]), int(t[1])), 2, (255, 0, 0), -1)
#        #cv2.imshow("s", img)
#        #cv2.waitKey(0) 
#
#    #for t in targets:
#    #    bbox = t['bbox']
#    #    category_id = t['category_id']
#    #    cv2.putText(img, str(category_id), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
#    #    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 1)

    