import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import HRNet
from dataset import COCOWholebody_BodyWithFeet
from util import JointsMSELoss, accuracy

batch_size = 6
device = "cuda"
print_freq = 100
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_annopath = "data\\annotations\\person_keypoints_wholebody_train.json"
train_imagepath = "data\\train2017"
train_dataset = COCOWholebody_BodyWithFeet(train_annopath, train_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize]))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True)

val_annopath = "data\\annotations\\person_keypoints_wholebody_val.json"
val_imagepath = "data\\val2017"
val_dataset = COCOWholebody_BodyWithFeet(val_annopath, val_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize]))
val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
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




best_train_acc = 0.0
best_val_acc = 0.0
for epoch in range(0, 5000):
    print("epoch : " + str(epoch))

    model.train()
    lr_scheduler.step()
    train_acc_list = []
    for i, (input, targets, target_weights, joints, joint_vis) in enumerate(train_loader):
        targets = targets.cuda(non_blocking=True)
        target_weights = target_weights.cuda(non_blocking=True)
        
        outputs = model(input)
        loss = criterion(outputs, targets, target_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                         targets.detach().cpu().numpy())
        train_acc_list.append(avg_acc)
        
        if (i + 1) % print_freq == 0:
            print("training... [{} / {}]( loss : {}, accuracy : {})".format(len(train_loader), i + 1, loss.item(), avg_acc))

    train_acc = sum(train_acc_list) / len(train_acc_list)
    if train_acc > best_train_acc:
        best_train_acc = train_acc
    print("average train accuracy : {}, best train accuracy : {})".format(train_acc, best_train_acc))

    model.eval()
    with torch.no_grad():
        val_acc_list = []
        for i, (input, targets, target_weights, joints, joint_vis) in enumerate(val_loader):
            targets = targets.cuda(non_blocking=True)
            target_weights = target_weights.cuda(non_blocking=True)

            outputs = model(input)
            loss = criterion(outputs, targets, target_weights)

            _, avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                             targets.detach().cpu().numpy())
            val_acc_list.append(avg_acc)

            if (i + 1) % print_freq == 0:
                print("validating... [{} / {}]( loss : {}, accuracy : {})".format(len(val_loader), i + 1, loss.item(), avg_acc))

        val_acc = sum(val_acc_list) / len(val_acc_list)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.module.state_dict(), "best_val_acc.pth")
        print("average val accuracy : {}, best val accuracy : {})".format(val_acc, best_val_acc))

    #torch.save(model.module.state_dict(), "final_state.pth")