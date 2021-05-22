#!/usr/bin/env python
# coding: utf-8

# 安装sklearn  
# 前提准备:  
#     安装numpy+mkl  
# - 按照numpy+mkl->scipy->sklearn的顺序一步步在cmd下 pip install+包的绝对路径 即可，这回再次import sklearn做测试，发现不会报错说你没有哪个哪个包了（No module named xxx）    
#   https://www.cnblogs.com/lyr2015/p/7891069.html  
#   文件下载:https://blog.csdn.net/qq_16725749/article/details/89396438

# numpy   1.19.5
# scipy   1.2.1

# In[7]:


import os
os.getcwd()


# In[ ]:


# import tensorflow as tf
# import keras
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


# In[6]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Seven'
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from face_dataset import load_dataset, resize_image, IMAGE_SIZE
import warnings
warnings.filterwarnings('ignore')
import os
import sys


# In[ ]:


import keras as k
k.__version__


# In[8]:


class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None
        # 验证集
        # self.valid_images = None
        # self.valid_labels = None
        # 测试集
        self.test_images = None
        self.test_labels = None
        # 数据加载路径
        self.path_name = path_name
        # 当前库采用的维度顺序
        self.input_shape = None
 
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2):
        # 加载数据集至内存
        images, labels = load_dataset(self.path_name)
        #train_labels和test_label只有一个标签1
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3,
                                                                                random_state=random.randint(0, 10))

        print('训练数据维度:',train_images.shape)#
# set_image_dim_ordering(dim_ordering)是用于设置图像的维度顺序的，有2个可选参数：
# （1）‘th’：即Theano模式，会把通道维放在第二个位置上。
# （2）‘tf’：即TensorFlow模式，会把通道维放在最后的位置上。
# 例：100张RGB三通道的16×32（高为16宽为32）彩色图
# th模式下的形式是（100, 3, 16, 32）分别是样本维100张图片、通道维3（颜色通道数）、高、宽
# tf模式下的形式是（100, 16, 32, 3）

#         if K.image_dim_ordering() == 'th':#版本问题报错,修改如下
        if K.image_data_format() == 'channels_first':#此处要仔细看一下通道维度位置变化对下面的影响
# 该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。
# 以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。
# 该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)#(样本,通道.长,宽)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)#同上
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)
 
            # 输出训练集、测试集的数量
            print(train_images.shape[0], 'train samples')
            print(test_images.shape[0], 'test samples')
            
            
            
            # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
            # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)
            # 像素数据浮点化以便归一化
            train_images = train_images.astype('float32')
            test_images = test_images.astype('float32')
            # 将其归一化,图像的各像素值归一化到0~1区间
            train_images /= 255.0
            test_images /= 255.0
            self.train_images = train_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.test_labels = test_labels
 

 
 
 
 


# In[9]:



# CNN网络模型类
class Model:
   def __init__(self):
       self.model = None

   # 建立模型
   def build_model(self, dataset, nb_classes=2):
       # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
       self.model = Sequential()

       # 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
       self.model.add(Conv2D(32, 3, 3, border_mode='same',
                                    input_shape=dataset.input_shape))  #  2维卷积层
       self.model.add(Activation('relu'))  #  激活函数层

       self.model.add(Conv2D(32, 3, 3))  # 2维卷积层
       self.model.add(Activation('relu'))  #  激活函数层

       self.model.add(MaxPool2D(pool_size=(2, 2)))  #  池化层
       self.model.add(Dropout(0.25))  #  Dropout层

       self.model.add(Conv2D(64, 3, 3, border_mode='same'))  #   2维卷积层
       self.model.add(Activation('relu'))  #  激活函数层

       self.model.add(Conv2D(64, 3, 3))  #  2维卷积层
       self.model.add(Activation('relu'))  #  激活函数层

       self.model.add(MaxPool2D(pool_size=(2, 2)))  #  池化层
       self.model.add(Dropout(0.25))  # Dropout层

       self.model.add(Flatten())  #  Flatten层
       self.model.add(Dense(512))  #  Dense层,又被称作全连接层
       self.model.add(Activation('relu'))  #  激活函数层
       self.model.add(Dropout(0.5))  # Dropout层
       self.model.add(Dense(nb_classes))  #  Dense层
       self.model.add(Activation('softmax'))  #  分类层，输出最终结果

       # 输出模型概况
       self.model.summary()

   # 训练模型
   def train(self, dataset, batch_size=20, nb_epoch=100, data_augmentation=True):
       sgd = SGD(lr=0.01, decay=1e-6,
                 momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
       self.model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])  # 完成实际的模型配置工作

       # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
       # 训练数据，有意识的提升训练数据规模，增加模型训练量
       if not data_augmentation:
           self.model.fit(dataset.train_images,
                          dataset.train_labels,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=(dataset.test_images, dataset.test_labels),
                          shuffle=True)
       # 使用实时数据提升
       else:
           # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
           # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
           datagen = ImageDataGenerator(
               featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
               samplewise_center=False,  # 是否使输入数据的每个样本均值为0
               featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
               samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
               zca_whitening=False,  # 是否对输入数据施以ZCA白化
               rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
               width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
               height_shift_range=0.2,  # 同上，只不过这里是垂直
               horizontal_flip=True,  # 是否进行随机水平翻转
               vertical_flip=False)  # 是否进行随机垂直翻转

           # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
           datagen.fit(dataset.train_images)

           # 利用生成器开始训练模型
           self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                 batch_size=batch_size),
                                    samples_per_epoch=dataset.train_images.shape[0],
                                    nb_epoch=nb_epoch,
                                    validation_data=(dataset.test_images, dataset.test_labels))

   MODEL_PATH = './mode_h5/face.model.h5'

   def save_model(self, file_path=MODEL_PATH):
       self.model.save(file_path)

   def load_model(self, file_path=MODEL_PATH):
       self.model = load_model(file_path)

   def evaluate(self, dataset):
       score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
       # print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
       print(f'{self.model.metrics_names[1]}:{score[1] * 100}%')

   # 识别人脸
   def face_predict(self, image):
       # 依然是根据后端系统确定维度顺序
       if K.image_data_format() == 'channels_first' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
           image = resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
           image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
       elif K.image_data_format() == 'channels_last' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
           image = resize_image(image)
           image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

           # 浮点并归一化
       image = image.astype('float32')
       image /= 255.0

       # 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
       result = self.model.predict_proba(image)
       print('result:', result)

       # 给出类别预测：0或者1
       result = self.model.predict_classes(image)

       # 返回类别预测结果
       return result[0]



if __name__ == '__main__':
    dataset = Dataset(os.getcwd())

    dataset.load()
 
    # 训练模型，这段代码不用，注释掉
    model = Model()
    model.build_model(dataset)
    # 测试训练函数的代码
    model.train(dataset)
	# model.save_model(file_path='./model_h5/me.face.model.h5')
    


# In[ ]:


# 严重过拟合


# In[ ]:

# 评估模型

# model = Model()
# # model.load_model(file_path='./model/me.face.model.h5')
# model.load_model(r"C:\Users\6\Desktop\objectForPClass\mode_h5\me.face.model.h5")
# model.evaluate(dataset)



