import os
import torch.utils.data as data
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import time
import copy

from src.model.models import Classifier

#数据预处理
data_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])]
)

NUM_EPOCH = 10  # 默认迭代次数
batch_size = 128  # 约占 10G 显存
device = torch.device('cuda:0')  # 默认使用 GPU
NUMCLASS = 10  # 类别数


#下载训练集-MNIST手写数字训练集
train_data = datasets.MNIST(root="./dataset", train=True, transform=data_tf, download=True)
test_data = datasets.MNIST(root="./dataset", train=False, transform=data_tf)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

dataloders = {
    "Train": trainloader, "Test": trainloader
}

total_train = len(train_data)


def train_model(model, crtiation, optimizer, schedular, num_epochs=NUM_EPOCH):
    save_path = "covnet_weights_fine_tune-with-train_val_tmp.pth"
    begin_time = time.time()
    best_weights = copy.deepcopy(model.state_dict())  # copy the weights from the model
    best_acc = 0.0
    arr_acc = []  # 用于作图
    # log_recode = []

    for epoch in range(num_epochs):
        print("-*-" * 20)
        item_acc = []
        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_acc = 0.0

            for images, labels in dataloders[phase]:
                images.to(device)
                labels.to(device)
                optimizer.zero_grad()
                # print(images.shape)
                # print(labels.shape)
                # exit()

                with torch.set_grad_enabled(phase == 'Train'):
                    opt = model(images.cuda())
                    # opt = model(images)
                    _, pred = torch.max(opt, 1)
                    labels = labels.cuda()
                    loss = crtiation(opt, labels)
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_acc += torch.sum(pred == labels)
            epoch_loss = running_loss / total_train
            epoch_acc = running_acc.double() / total_train
            print('epoch={}, Phase={}, Loss={:.4f}, ACC:{:.4f}'.format(epoch, phase,
                                                                       epoch_loss, epoch_acc))
            # log_recode.append((epoch, phase, epoch_loss, epoch_acc))
            item_acc.append(epoch_acc)

            if phase == "Train":
                schedular.step()
            if phase == 'Test' and epoch_acc > best_acc:
                # Upgrade the weights
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
        arr_acc.append(item_acc)
        time_elapes = int(time.time() - begin_time)
        print("use time: {:2d}:{:2d}".format(
            time_elapes // 60, time_elapes % 60
        ))

    print('Best Val ACC: {:}'.format(best_acc))
    torch.save(best_weights, save_path)
    # with open("fine_tune_covnet_out.log") as fp:
    #     for e, p, l, a in log_recode:
    #         fp.write('epoch={}, Phase={}, Loss={:.4f}, ACC:{:.4f}'.format(e, p, l, a))
    model.load_state_dict(best_weights)  # 保存最好的参数
    return model, arr_acc


def main():
    model_ft = Classifier(layer_size=64, num_channels=1, nClasses=NUMCLASS, image_size=28)
    model_ft = model_ft.to(device)
    model_ft.cuda()
    print(model_ft)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.005, betas=(0.5, 0.999))
    # optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    model_ft, arr_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, NUM_EPOCH)


if __name__ == '__main__':
    main()
