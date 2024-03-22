# Minimalistic-HRNet
 
## Introduction
This repo is a lightweight pytorch implementation of the paper : [High-Resolution Representations for Labeling Pixels and Regions](https://arxiv.org/abs/1904.04514 "link")

The official pytorch implementation is here : [HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation "link")

HRNet's model structure and details can be found in the the link above. The main goal of this repo is to make HRNet as easy to use as possible and lightweight enough to implement in any project.

The dataset I'm using is [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody "link"). The dataset in total has 133 keypoints (17 for body, 6 for feet, 68 for face and 42 for hands). The original COCO Keypoints 2017 dataset only has 17 body keypoints, lacks of hands, face, and feet keypoints. The training detail will be elaborate further down below.

The repo has been tested in :
```
Windows 11
Python 3.8
Pytorch 1.9.1
```


## Installation
Install all the dependencies.

I recommend using a venv manager like Conda to setup your python environment.
```
pip install -r requirements.txt
```



Make three folders in root folder and name them  **data**, **log** and **weight**, then you should have a directory tree like this : 

```
${Minimalistic-HRNet root}
├── data
├── dataset
├── log
├── model
├── util
├── weight
├── .gitattributes 
├── .gitignore
├── joint_test.py
├── joint_train.py
├── LICENSE
├── live_demo.py
├── README.md
└── requirements.txt
```

Done !

## Training
### Data Preparation
Download **2017 Train images** and **2017 Val images** in [COCO 2017 website](https://cocodataset.org/#download "link")

Download **COCO-WholeBody annotations** in [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody "link")

Make a folder in root folder and name it **data**,

and put image folders and annotations inside the **data** folder.

Directory tree inside **data** folder should look like this.
```
${Minimalistic-HRNet root}
├── data
    ├── annotations
        ├── coco_wholebody_train_v1.0.json
        ├── coco_wholebody_val_v1.0.json
    ├── train2017
        ├── train images...
    ├── val2017
        ├── val images...
...
```

### Configuration

Inside **joint_train.py** in root folder, has a config section like this : 
```python
######################################### Model training config start:
batch_size = 12
device = "cuda"

train_annopath = "data\\coco_wholebody\\annotations\\coco_wholebody_train_v1.0.json"
train_imagepath = "data\\coco_wholebody\\train2017"
val_annopath = "data\\coco_wholebody\\annotations\\coco_wholebody_val_v1.0.json"
val_imagepath = "data\\coco_wholebody\\val2017"

resume_training = False
model_save_path = "weight/latest.pth"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset = COCOWholebody_BodyWithFeetAndPalm(train_annopath, train_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize]),
                                                    image_height=384, image_width=288, heatmap_height=96, heatmap_width=72)
val_dataset = COCOWholebody_BodyWithFeetAndPalm(val_annopath, val_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize]),
                                                    image_height=384, image_width=288, heatmap_height=96, heatmap_width=72)

lr = 0.001
lr_step = [100, 170]
lr_factor = 0.1
base_channels = 48
out_channels = train_dataset.num_joints
######################################### Model training config end.
```
This is the place where you can define model's training parameter.

If you want to start training, save the training parameters and simply run **joint_train.py** : 

```
python joint_train.py
```

The training heatmap log will be saved into the **log** folder.

And weight file(.pth) will be saved into the **weight** folder.

## Testing
```
python joint_test.py
```

## Live Detection
```
python live_demo.py
```