import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import HRNet
from dataset import COCOWholebody_BodyWithFeet
from loss import JointsMSELoss


train_annopath = "data\\annotations\\person_keypoints_wholebody_val.json"
train_imagepath = "data\\val2017"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transforms.Compose([transforms.ToTensor(), normalize])
train_dataset = COCOWholebody_BodyWithFeet(train_annopath, train_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize]))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=6,
                                           shuffle=True,
                                           num_workers=32,
                                           pin_memory=True)

lr = 0.001
lr_step = [170, 200]
lr_factor = 0.1
last_epoch = -1
model = HRNet(out_channels=23)
model = torch.nn.DataParallel(model, device_ids=(0,)).cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    lr_step, 
                                                    lr_factor,
                                                    last_epoch=last_epoch)

criterion = JointsMSELoss(use_target_weight=True).cuda()

model.train()
for epoch in range(0, 5000):
    lr_scheduler.step()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        
        outputs = model(input)

        loss = 0.0
        for output in outputs:
            loss += criterion(output, target, target_weight)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#input = torch.randn(1, 3, 256, 192)
#output = model.forward(input)
#print(output.size())