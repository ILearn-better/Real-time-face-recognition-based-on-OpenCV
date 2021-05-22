#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
os.getcwd()


# In[7]:


# os.chdir(os.path.join(os.getcwd(),'face_serve'))
os.getcwd()
# os.listdir(os.getcwd())


# Python endswith() 方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False。
# 可选参数"start"与"end"为检索字符串的开始与结束位置。

# 
# _____________________________________________________________________________________________________________

#  - os.path.isdir用于判断某一对象(需提供绝对路径)是否为目录
#  - os.path.isfile()用于判断某一对象(需提供绝对路径)是否为文件
#  - os.path.abspath()获取文件的完整路径(绝对路径)

# In[8]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Seven'
import os
import numpy as np
import cv2
 
# 定义图片尺寸
IMAGE_SIZE = 64

# 按照定义图像大小进行尺度调整
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = 0, 0, 0, 0
    # 获取图像尺寸
    h, w, _ = image.shape
    # 找到图片最长的一边
    longest_edge = max(h, w)
    # 计算短边需要填充多少使其与长边等长
    if h < longest_edge:
        d = longest_edge - h
        top = d // 2
        bottom = d // 2
    elif w < longest_edge:
        d = longest_edge - w
        left = d // 2
        right = d // 2
    else:
        pass
 
    # 设置填充颜色
    BLACK = [0, 0, 0]
    # 对原始图片进行填充操作
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # 调整图像大小并返回
    return cv2.resize(constant, (height, width))
 
images, labels = list(), list()
# 读取训练数据
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        # 合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        # 如果是文件夹，则继续递归调用
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.jpg'):
                # print(dir_item)
                image = cv2.imread(full_path)#此处路径斜杠问题会引起报错
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                images.append(image)
                labels.append(path_name)
#                 print(labels)#此处输出为path_name路径list,表明当前数据集合标签为此
#     print(path_name)
    return images, labels

# 从指定路径+读取训练数据
def load_dataset(path_name):
    images, labels = read_path(path_name)
    # 由于图片是基于矩阵计算的， 将其转为矩阵
#     print(labels)
    images = np.array(images)
#     print(images.shape)
    labels = np.array([0 if label.endswith('0') else 1 for label in labels])##此处根据实际改判断条件
    return images, labels

if __name__ == '__main__':
    images, labels = load_dataset(os.getcwd())
    print('load over')


# In[9]:


# from sklearn.model_selection import train_test_split
# import random

# train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3,
#                                                                         random_state=random.randint(0, 10))


# In[11]:


# set(labels)

