#!/usr/bin/env python

import os
import paddle
from paddle import nn

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import cv2


class MyDateset(paddle.io.Dataset):
    def __init__(
        self,
        mode='train',
        train_imgs_dir='./data/rawdata/train_datasets_document_detection_0411/images/',
        train_txt='./data/rawdata/train_datasets_document_detection_0411/data_info.txt'
    ):
        super(MyDateset, self).__init__()

        self.mode = mode
        self.train_imgs_dir = train_imgs_dir

        with open(train_txt, 'r') as f:
            self.train_infor = f.readlines()

    def __getitem__(self, index):
        item = self.train_infor[index][:-1]
        splited = item.split(',')
        img_name = splited[0]

        img = cv2.imread(self.train_imgs_dir + img_name + '.jpg')
        h, w, c = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gray_scale = paddle.vision.transforms.Grayscale(num_output_channels=3)
        img = gray_scale(img)

        # 对图片进行resize，调整明暗对比度等参数
        img_size = 512
        # if np.random.rand() < 1 / 3:
        #     img_size =  512
        # elif np.random.rand() < 2 / 3:
        #     img_size = 384
        # else:
        #     img_size = 768

        img = paddle.vision.transforms.resize(img, (img_size, img_size),
                                              interpolation='bilinear')

        if np.random.rand() < 1 / 3:
            # horizon flip
            img[:, ::-1, :] = img
            for i in range(1, len(splited), 2):
                splited[i] = w - float(splited[i])
                splited[i + 1] = float(splited[i + 1])
        elif np.random.rand() < 2 / 3:
            # vertical flip
            img[::-1, :, :] = img
            for i in range(1, len(splited), 2):
                splited[i] = float(splited[i])
                splited[i + 1] = h - float(splited[i + 1])

        r_x = np.random.rand()
        if r_x < 1 / 3:
            img = paddle.vision.transforms.adjust_brightness(
                img,
                np.random.rand() * 2)
        elif r_x < 2 / 3:
            if np.random.rand() < 1 / 2:
                img = paddle.vision.transforms.adjust_contrast(
                    img,
                    np.random.rand() * 2)
            else:
                img = paddle.vision.transforms.adjust_hue(
                    img,
                    np.random.rand() - 0.5)

        img = img.transpose((2, 0, 1))
        img = img / 255

        sites = []
        for i in range(1, len(splited), 2):
            sites.append([float(splited[i]) / w, float(splited[i + 1]) / h])

        label = []
        for i in range(4):
            x, y = self.get_corner(sites, i + 1)
            label.append(x)
            label.append(y)

        img = paddle.to_tensor(img).astype('float32')
        label = paddle.to_tensor(label).astype('float32')

        return img, label

    def get_corner(self, sites, corner_flag):
        # corner_flag 1:top_left 2:top_right 3:bottom_right 4:bottom_left
        if corner_flag == 1:
            target_sites = [0, 0]
        elif corner_flag == 2:
            target_sites = [1, 0]
        elif corner_flag == 3:
            target_sites = [1, 1]
        elif corner_flag == 4:
            target_sites = [0, 1]

        min_dis = 3
        best_x = 0
        best_y = 0
        for site in sites:
            if abs(site[0] - target_sites[0]) + abs(site[1] -
                                                    target_sites[1]) < min_dis:
                min_dis = abs(site[0] - target_sites[0]) + abs(site[1] -
                                                               target_sites[1])
                best_x = site[0]
                best_y = site[1]

        return best_x, best_y

    def __len__(self):
        return len(self.train_infor)


# 对dataloader进行测试
'''
train_dataset=MyDateset()

train_dataloader = paddle.io.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=False)

for step, data in enumerate(train_dataloader):
    img, label = data
    print(step, img.shape, label.shape)
    break

'''


class Encoder(nn.Layer):  #下采样：两层卷积，两层归一化，最后池化。
    def __init__(self, num_channels, num_filters):
        super(Encoder, self).__init__()  #继承父类的初始化
        self.conv1 = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=3,  #3x3卷积核，步长为1，填充为1，不改变图片尺寸[H W]
            stride=1,
            padding=1)
        self.bn1 = nn.BatchNorm(num_filters, act="relu")  #归一化，并使用了激活函数

        self.conv2 = nn.Conv2D(in_channels=num_filters,
                               out_channels=num_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm(num_filters, act="relu")

        self.pool = nn.MaxPool2D(kernel_size=2, stride=2,
                                 padding="SAME")  #池化层，图片尺寸减半[H/2 W/2]

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_conv = x  #两个输出，灰色 ->
        x_pool = self.pool(x)  #两个输出，红色 |
        return x_conv, x_pool


class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.resnet = paddle.vision.models.resnet152(pretrained=True,
                                                     num_classes=0)
        self.flatten = paddle.nn.Flatten()

        # self.mid_conv1 = nn.Conv2D(512, 2048, 1)  #中间层
        # self.mid_bn1 = nn.BatchNorm(2048, act="relu")
        # self.mid_conv2 = nn.Conv2D(2048, 1024, 1)
        # self.mid_bn2 = nn.BatchNorm(2048, act="relu")

        self.linear = paddle.nn.Linear(2048, 8)

    def forward(self, img):
        y = self.resnet(img)
        y = self.flatten(y)

        # y = self.mid_conv1(y)
        # y = self.mid_bn1(y)
        # y = self.mid_conv2(y)
        # y = self.mid_bn2(y)

        out = self.linear(y)

        return out


class NewNet(paddle.nn.Layer):
    def __init__(self):
        super(NewNet, self).__init__()
        self.resnet = paddle.vision.models.resnet152(pretrained=True,
                                                     num_classes=0)
        self.flatten = paddle.nn.Flatten()
        self.linear_1 = paddle.nn.Linear(2048, 512)
        self.linear_2 = paddle.nn.Linear(512, 256)
        self.linear_3 = paddle.nn.Linear(256, 8)

        # self.mid_conv1 = nn.Conv2D(512, 2048, 1)  #中间层
        # self.mid_bn1 = nn.BatchNorm(2048, act="relu")
        # self.mid_conv2 = nn.Conv2D(2048, 1024, 1)
        # self.mid_bn2 = nn.BatchNorm(2048, act="relu")

        # self.linear = paddle.nn.Linear(2048, 8)

        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.2)

    def forward(self, img):
        y = self.resnet(img)
        y = self.flatten(y)

        # y = self.mid_conv1(y)
        # y = self.mid_bn1(y)
        # y = self.mid_conv2(y)
        # y = self.mid_bn2(y)

        y = self.linear_1(y)
        y = self.linear_2(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_3(y)
        out = paddle.nn.functional.sigmoid(y)

        # out = self.linear(y)

        return out


from sklearn.metrics.pairwise import euclidean_distances


def cal_coordinate_loss(logit, label, alpha=0.5):
    mse_loss = nn.MSELoss(reduction='mean')

    mse_x0 = mse_loss(logit[:, 0], label[:, 0])
    mse_y0 = mse_loss(logit[:, 1], label[:, 1])
    mse_x1 = mse_loss(logit[:, 2], label[:, 2])
    mse_y1 = mse_loss(logit[:, 3], label[:, 3])
    mse_x2 = mse_loss(logit[:, 4], label[:, 4])
    mse_y2 = mse_loss(logit[:, 5], label[:, 5])
    mse_x3 = mse_loss(logit[:, 6], label[:, 6])
    mse_y3 = mse_loss(logit[:, 7], label[:, 7])

    mse_x = (mse_x0 + mse_x1 + mse_x2 + mse_x3) / 4
    mse_y = (mse_y0 + mse_y1 + mse_y2 + mse_y3) / 4
    mse_l = 0.5 * (mse_x + mse_y)

    ed_loss = []
    for i in range(logit.shape[0]):
        logit_tmp = logit[i, :].numpy()
        label_tmp = label[i, :].numpy()

        ed_tmp = euclidean_distances([logit_tmp], [label_tmp])
        ed_loss.append(ed_tmp)

    ed_l = sum(ed_loss) / len(ed_loss)

    loss = alpha * mse_l + (1 - alpha) * ed_l

    return loss


class MyLoss(paddle.nn.Layer):
    def __init(self):
        self(MyLoss, self).__init__()

    def forward(self, input, label):
        output = cal_coordinate_loss(input, label)
        return output


model = MyNet()
#  model = NewNet()
model.train()

train_dataset = MyDateset()

# 需要接续之前的模型重复训练可以取消注释
last_model_path = "./last_model.pdparams"
if os.path.exists(last_model_path):
    param_dict = paddle.load(last_model_path)
    model.load_dict(param_dict)

train_dataloader = paddle.io.DataLoader(train_dataset,
                                        batch_size=12,
                                        shuffle=True,
                                        drop_last=False)

visualdl = paddle.callbacks.VisualDL(log_dir="visual_log")

# max_epoch = 10
# max_epoch = 30
#max_epoch = 50
max_epoch = 100
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
    #  learning_rate=0.0001, # 0.84578
    #  learning_rate=1e-5,  # 0.87541
    # learning_rate=1e-6,  # 0.88823
    #  learning_rate=1e-6,  # continue train 0.89052
    # Multi Test 3: 0.88732 !!!bad!!!
    # learning_rate=1e-6,  # origin continue 0.89323
    # learning_rate=1e-6, # horz/vert flip 0.89502

    # learning_rate=1e-5,  # New Net 0.81218
    #  learning_rate=1e-5,  #  +100 epochs 0.81287
    # learning_rate=1e-6,  # lr=1e-6 + 100 epochs 0.8403
    #  learning_rate=1e-6,  # SmoothL1Loss lr=1e-6 + 100 epochs  0.83376
    #  learning_rate=1e-6,  # MyNet MyLoss lr=1e-6 + 100 epochs   0.00751

    #  learning_rate=1e-6,  # MyNet SmoothL1Loss lr=1e-6 + 100 epochs  0.90445
    learning_rate=1e-6,  # MyNet MyLoss lr=1e-6 + 100 epochs  0.90524
    T_max=max_epoch)
opt = paddle.optimizer.Adam(learning_rate=scheduler,
                            parameters=model.parameters())

checkpoints_dir = "./checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)
best_loss = float('inf')
now_step = 0
for epoch in trange(max_epoch):
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} train")
    for step, data in enumerate(pbar):
        now_step += 1

        img, label = data
        pre = model(img)

        # loss = paddle.nn.functional.square_error_cost(pre, label).mean()
        #  loss = MyLoss()(pre, label).mean()
        loss = paddle.nn.SmoothL1Loss()(pre, label).mean()

        loss.backward()
        opt.step()
        opt.clear_gradients()
        if now_step % 10 == 0:
            pbar.set_postfix({'loss': f"{loss.mean().numpy()}"})
            # print("epoch: {}, batch: {}, loss is: {}".format(
            #     epoch, step,
            #     loss.mean().numpy()))
        pbar.update(1)
    pbar.close()
    #  paddle.save(model.state_dict(),
    #              f'{checkpoints_dir}/model_{epoch}.pdparams')

paddle.save(model.state_dict(), 'model.pdparams')
