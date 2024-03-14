import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import HRNet
from dataset import COCOWholebody_BBox
from util import IOULoss
from util.bbox_util import *
import matplotlib.pyplot as plt
import numpy as np
import cv2

def train(model, train_dataset, train_loader, lr_scheduler, criterion, optimizer, epoch, print_freq):
    model.train()
    lr_scheduler.step()
    train_acc_list = []
    train_loss_list = []
    for i, (input, targets, data_idx) in enumerate(train_loader):
        targets = targets.cuda(non_blocking=True)
        
        outputs = model(input)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred, acc, avg_acc = accuracy(outputs.detach().cpu().numpy(), 
                                 targets.detach().cpu().numpy(), 
                                 train_dataset.image_width, 
                                 train_dataset.image_height)

        train_acc_list.append(avg_acc)
        train_loss_list.append(loss.item())
        
        if (i + 1) % print_freq == 0:
            print("training... [{} / {}]( loss : {}, accuracy : {})".format(len(train_loader), i + 1, loss.item(), avg_acc))
            get_output_result(train_dataset, outputs, targets, data_idx, "train", epoch, i + 1)

    train_acc = sum(train_acc_list) / len(train_acc_list)
    train_loss = sum(train_loss_list) / len(train_loss_list)

    return train_loss, train_acc

def val(model, val_dataset, val_loader, criterion, epoch, print_freq):
    model.eval()
    val_acc_list = []
    val_loss_list = []
    with torch.no_grad():
        for i, (input, targets, data_idx) in enumerate(val_loader):
            targets = targets.cuda(non_blocking=True)

            outputs = model(input)
            loss = criterion(outputs, targets)

            _, _, avg_acc = accuracy(outputs.detach().cpu().numpy(), 
                                     targets.detach().cpu().numpy(), 
                                     val_dataset.image_width, 
                                     val_dataset.image_height)

            val_acc_list.append(avg_acc)
            val_loss_list.append(loss.item())

            if (i + 1) % print_freq == 0:
                print("validating... [{} / {}]( loss : {}, accuracy : {})".format(len(val_loader), i + 1, loss.item(), avg_acc))
                get_output_result(val_dataset, outputs, targets, data_idx, "val", epoch, i + 1)

    val_acc = sum(val_acc_list) / len(val_acc_list)
    val_loss = sum(val_loss_list) / len(val_loss_list)

    return val_loss, val_acc

def get_output_result(dataset, outputs, targets, data_idx, image_name, epoch, batch_num):
    img_output = []
    for i, (output, target, data_id) in enumerate(zip(outputs, targets, data_idx)):
        img = dataset.get_preprocessed_image(data_id)
        resized_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 2
        #img_concat = resized_gray
        target_concat = resized_gray
        output_concat = resized_gray
        for t, ta in zip(output, target):
            t = t.detach().cpu().numpy()
            ta = ta.detach().cpu().numpy()
            t = cv2.resize(t, (img.shape[1], img.shape[0]))
            ta = cv2.resize(ta, (img.shape[1], img.shape[0]))
            t *= 255 / 2
            ta *= 255 / 2
            t += resized_gray
            ta += resized_gray
            #img_concat = np.hstack((img_concat, resized_gray))
            target_concat = np.hstack((target_concat, ta))
            output_concat = np.hstack((output_concat, t))
        if i == 0:
            img_output = np.vstack((target_concat, output_concat))
        else:
            img_output = np.vstack((img_output, target_concat))
            img_output = np.vstack((img_output, output_concat)) 
    plt.imshow(img_output, vmin=0, vmax=255)
    plt.savefig('log/' + image_name + '_' + str(epoch) + '_' + str(batch_num) + '.png', dpi=500)

batch_size = 16
device = "cuda"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_annopath = "data\\annotations\\person_keypoints_wholebody_train.json"
train_imagepath = "data\\train2017"
train_dataset = COCOWholebody_BBox(train_annopath, train_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize]),
                                           image_height=288, image_width=384, heatmap_height=72, heatmap_width=96)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True)

val_annopath = "data\\annotations\\person_keypoints_wholebody_val.json"
val_imagepath = "data\\val2017"
val_dataset = COCOWholebody_BBox(val_annopath, val_imagepath, transforms=transforms.Compose([transforms.ToTensor(), normalize]),
                                         image_height=288, image_width=384, heatmap_height=72, heatmap_width=96)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True)

lr = 0.001
lr_step = [170, 200]
lr_factor = 0.1
last_epoch = -1
model = HRNet(out_channels=4)
model.load_state_dict(torch.load("weight/latest.pth"))
model = torch.nn.DataParallel(model, device_ids=(0,)).cuda()


optimizer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    lr_step, 
                                                    lr_factor,
                                                    last_epoch=last_epoch)

criterion = IOULoss().cuda()

best_train_acc = 0.0
best_val_acc = 0.0
best_train_loss = float('inf')
best_val_loss = float('inf')
for epoch in range(0, 5000):
    print("epoch : " + str(epoch))

    train_loss, train_acc = train(model, train_dataset, train_loader, lr_scheduler, criterion, optimizer, epoch, 200)
    if train_acc > best_train_acc:
        best_train_acc = train_acc
    if train_loss < best_train_loss:
        best_train_loss = train_loss
    print("best train loss : {}, best train accuracy : {})".format(best_train_loss, best_train_acc))

    val_loss, val_acc = val(model, val_dataset, val_loader, criterion, epoch, 20)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.module.state_dict(), "weight/best_acc.pth")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.module.state_dict(), "weight/best_loss.pth")
    print("best val loss : {}, best val accuracy : {})".format(best_val_loss, best_val_acc)) 

    torch.save(model.module.state_dict(), "weight/latest.pth")