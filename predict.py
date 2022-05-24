#!/usr/bin/env python

# 代码示例
# python predict.py [src_image_dir] result

import os
import sys
import glob
import json
import cv2
import paddle
import numpy as np
import pandas as pd
#  from tqdm import tqdm


class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.resnet = paddle.vision.models.resnet152(pretrained=True,
                                                     num_classes=0)
        self.flatten = paddle.nn.Flatten()
        self.linear = paddle.nn.Linear(2048, 8)

    def forward(self, img):
        y = self.resnet(img)
        y = self.flatten(y)
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


def process(src_image_dir, save_dir):
    model = MyNet()
    #  model = NewNet()
    param_dict = paddle.load('./model.pdparams')
    model.load_dict(param_dict)
    json_results = []
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    #  for image_path in tqdm(image_paths):
    for image_path in image_paths:

        def predict_one(img):
            # do something
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_size = 512
            # if np.random.rand() < 1 / 3:
            #     img_size =  512
            # elif np.random.rand() < 2 / 3:
            #     img_size = 384
            # else:
            #     img_size = 768
            img = paddle.vision.transforms.resize(img, (img_size, img_size),
                                                  interpolation='bilinear')

            # if np.random.rand() < 1 / 3:
            #     img = paddle.vision.transforms.adjust_brightness(
            #         img,
            #         np.random.rand() * 2)
            # else:
            #     if np.random.rand() < 1 / 2:
            #         img = paddle.vision.transforms.adjust_contrast(
            #             img,
            #             np.random.rand() * 2)
            #     else:
            #         img = paddle.vision.transforms.adjust_hue(
            #             img,
            #             np.random.rand() - 0.5)

            img = img.transpose((2, 0, 1))
            img = img / 255

            img = paddle.to_tensor(img).astype('float32')
            img = img.reshape([1] + img.shape)
            pre = model(img)[0].numpy()

            return pre

        num_predict = 1
        all_pre = []
        img = cv2.imread(image_path)
        h, w, c = img.shape
        for k in range(num_predict):
            pre = predict_one(img)
            all_pre.append(pre)
        pre = np.mean(all_pre, axis=0)

        # print(f"all_pre: {all_pre}")
        # print(f"pre: {pre}")
        # raise ValueError('')

        points = []
        for i in range(0, 8, 2):
            points.append({'x': pre[i] * w, 'y': pre[i + 1] * h})

        json_results.append({
            'image_id': image_path.split('/')[-1],
            'points': points
        })

        # 保存结果图片
        # save_path = os.path.join(save_dir, os.path.basename(image_path).replace(".jpg", ".png"))
        # cv2.imwrite(save_path, out_image)

    # 或者保存坐标信息到json文件
    json_data = {"results": json_results}
    with open("{}/result.json".format(save_dir), "w") as fid:
        fid.write(json.dumps(json_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process(src_image_dir, save_dir)

# 压缩可提交文件
# ! zip submit.zip model.pdparams predict.py
