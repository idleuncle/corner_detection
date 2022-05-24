# 百度网盘AI大赛-文档检测优化赛第19名方案



比赛官网： https://aistudio.baidu.com/aistudio/competition/detail/207/0/introduction

Baseline: https://aistudio.baidu.com/aistudio/projectdetail/3861946

## 赛题任务

生活中人们使用手机进行文档扫描逐渐成为一件普遍的事情，为了提高人们的使用体验，我们期望通过算法技术去除杂乱的拍摄背景并精准框取文档边缘，选手需要通过深度学习技术训练模型，对给定的真实场景下采集得到的带有拍摄背景的文件图片进行边缘智能识别，并最终输出处理后的扫描结果图片。



## 数据集描述

在本次比赛最新发布的数据集中，大部分图像数据均是通过真实场景拍摄采集的，小部分是通过网络公开数据收集的。该任务的输入为文档类型的图像，期望输出文档图像四个角的坐标点，由于不是所有的文档图像都是规则的四边形，因此本次比赛提供的训练数据中包括四个部分：文档图像、文档边缘坐标点、预生成的边缘heatmap图、预生成的文档区域分割图，其中，图像四个角的坐标点可通过文档边缘坐标点来生成；发布的数据集GT形式较多，是为了不限制大家完成该任务的思路。另外，为达到更好的算法效果，本次比赛不限制大家使用额外的训练数据来优化模型。
测试数据集的GT不做公开，请各位选手基于本次比赛最新发布的测试数据集提交模型以及代码，后台会自动运行评测脚本。



## 数据集构成

```
|- root
    |- images
    |- edges
    |- segments
    - data_info.txt
```

- 本次比赛最新发布的数据集共包含训练集、A榜测试集、B榜测试集三个部分，其中，训练集共2797个样本，A榜测试集共597个样本，B榜测试集共606个样本；
- images 为文档图像数据，edges 为预生成的边缘heatmap图，segments 为预生成的文档区域分割图，根据图片名称一一对应；
- data_info.txt 文件中的每一行对应一个图像样本，其数据格式如下： 图片名称,x1,y1,x2,y2,x3,y3,…,xn,yn

 

## 训练



```python
# make train
python train.py
```

超参调整：lr=1e-6, batch_size=12 (V100 16G)

数据增强：图像随机水平垂直翻转、图片亮度对比度随机变换

训练过程：基于预训练模型resnet152训练100 epoch，重复加载上一轮模型权重（ln -s mode.pdparams last_model.pdparams）训练的100 epoch共3轮。

得分提升：baseline: 0.84578, final: 0.90524



## 推理

```python
# make predict
 python predict.py \
        ./data/rawdata/testA_datasets_document_detection/images \
        ./results
```

模型权重链接: https://pan.baidu.com/s/18IDmC_KMsqryX4FQN5e-8g?pwd=dibz 提取码: dibz 



## 打包上传结果



```python
# make submit
zip submit.zip model.pdparams predict.py
```

