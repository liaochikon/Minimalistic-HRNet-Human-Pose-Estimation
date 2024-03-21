import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from model import HRNet
from dataset import COCOWholebody_BodyWithFeetAndPalm
from util.joint_loss import JointsMSELoss
from util.joint_util import accuracy

def train(model, train_dataset, train_loader, criterion, optimizer, epoch, print_freq):
    model.train()
    train_acc_list = []
    train_loss_list = []
    for i, (input, targets, target_weights, data_idx, flipped) in enumerate(train_loader):
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
        train_loss_list.append(loss.item())
        
        if (i + 1) % print_freq == 0:
            print("training... [{} / {}]( loss : {}, accuracy : {})".format(len(train_loader), i + 1, loss.item(), avg_acc))
            get_output_result(train_dataset, flipped, outputs, targets, data_idx, "train", epoch, i + 1)

    train_acc = sum(train_acc_list) / len(train_acc_list)
    train_loss = sum(train_loss_list) / len(train_loss_list)

    return train_loss, train_acc

def val(model, val_dataset, val_loader, criterion, epoch, print_freq):
    model.eval()
    val_acc_list = []
    val_loss_list = []
    with torch.no_grad():
        for i, (input, targets, target_weights, data_idx, flipped) in enumerate(val_loader):
            targets = targets.cuda(non_blocking=True)
            target_weights = target_weights.cuda(non_blocking=True)

            outputs = model(input)
            loss = criterion(outputs, targets, target_weights)

            _, avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                             targets.detach().cpu().numpy())
            val_acc_list.append(avg_acc)
            val_loss_list.append(loss.item())

            if (i + 1) % print_freq == 0:
                print("validating... [{} / {}]( loss : {}, accuracy : {})".format(len(val_loader), i + 1, loss.item(), avg_acc))
                get_output_result(val_dataset, flipped, outputs, targets, data_idx, "val", epoch, i + 1)

    val_acc = sum(val_acc_list) / len(val_acc_list)
    val_loss = sum(val_loss_list) / len(val_loss_list)

    return val_loss, val_acc

def get_output_result(dataset, flipped, outputs, targets, data_idx, image_name, epoch, batch_num):
    img_output = []
    for i, (output, target, data_id) in enumerate(zip(outputs, targets, data_idx)):
        img = []
        if flipped[i]:
            img = dataset.get_filpbody_image(data_id)
        else:
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

######################################### Model training config start:
batch_size = 12
device = "cuda"

train_annopath = "data\\annotations\\person_keypoints_wholebody_train.json"
train_imagepath = "data\\train2017"
val_annopath = "data\\annotations\\person_keypoints_wholebody_val.json"
val_imagepath = "data\\val2017"

resume_training = True
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

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=True)

    model_dict = {"epoch" : -1,
                  "train_loss_list" : [], 
                  "train_accuracy_list" : [], 
                  "val_loss_list" : [], 
                  "val_accuracy_list" : [], 
                  "best_train_loss" : float('inf'), 
                  "best_train_accuracy" : 0.0, 
                  "best_val_loss" : float('inf'), 
                  "best_val_accuracy" : 0.0,
                  "model_state_dict": None,
                  "optimizer_state_dict": None,}

    model = HRNet(base_channels=base_channels ,out_channels=out_channels)
    criterion = JointsMSELoss(use_target_weight=True).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if resume_training:
        model_dict = torch.load(model_save_path)
        model.load_state_dict(model_dict['model_state_dict'])
        model = torch.nn.DataParallel(model, device_ids=(0,)).cuda()
        optimizer.load_state_dict(model_dict['optimizer_state_dict'])
    else:
        model = torch.nn.DataParallel(model, device_ids=(0,)).cuda()

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                        lr_step, 
                                                        lr_factor,
                                                        last_epoch=model_dict['epoch'])

    if model_dict['epoch'] == -1:
        model_dict['epoch'] = 0
    for epoch in range(model_dict['epoch'], 5000):
        print("epoch : " + str(epoch))
        print("current learn rate : {}".format(lr_scheduler.get_last_lr()))
        model_dict['epoch'] = epoch

        train_loss, train_acc = train(model, train_dataset, train_loader, criterion, optimizer, epoch, 300)
        model_dict['model_state_dict'] = model.module.state_dict()
        model_dict['optimizer_state_dict'] = optimizer.state_dict()

        if train_acc > model_dict['best_train_accuracy']:
            model_dict['best_train_accuracy'] = train_acc
        if train_loss < model_dict['best_train_loss']:
            model_dict['best_train_loss'] = train_loss

        model_dict['train_loss_list'].append(train_loss)
        model_dict['train_accuracy_list'].append(train_acc)
        print("current train loss : {}, current train accuracy : {})".format(train_loss, train_acc))
        print("best train loss : {}, best train accuracy : {})".format(model_dict['best_train_loss'], model_dict['best_train_accuracy']))

        val_loss, val_acc = val(model, val_dataset, val_loader, criterion, epoch, 40)
        if val_acc > model_dict['best_val_accuracy']:
            model_dict['best_val_accuracy'] = val_acc
            torch.save(model_dict, "weight/best_acc.pth")
        if val_loss < model_dict['best_val_loss']:
            model_dict['best_val_loss'] = val_loss
            torch.save(model_dict, "weight/best_loss.pth")

        print("current val loss : {}, current val accuracy : {})".format(val_loss, val_acc))
        print("best val loss : {}, best val accuracy : {})".format(model_dict['best_val_loss'], model_dict['best_val_accuracy']))
        model_dict['val_loss_list'].append(val_loss)
        model_dict['val_accuracy_list'].append(val_acc)

        torch.save(model_dict, "weight/latest.pth")
        lr_scheduler.step()