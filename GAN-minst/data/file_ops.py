#！/usr/bin/env python
#-*- coding: utf-8 -*-

'''
data file deal for train/test/vali  API
'''

import os
import glob
import random
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from macpath import join
import sys
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
  xrange
except:
  xrange = range

class Config():
  def __init__(self):
    '''
    训练 I-LR：(256*256*3)  from 标签I-HR：(4608*3328*3) crop (256*256*3) 按1-234 比例产生子子集
    目前样本总数为: 16张，生成训练样本数： 3744张
    '''
    self.is_train = True
    
    self.image_w = 256
    self.image_h = 256
    self.tw = 3328
    self.th = 4608 
    self.train_data = "traindata"
    self.test_data = "testdata"
    self.train_data_tfrecord = "train-tfrecord"
    self.test_data_tfrecord = "test-tfrecord"

    self.index = 0
   
class Data():
  
  def __init__(self):
    pass

  def data_list(self,config):
    """返回图片文件列表"""
    if config.is_train: 
      data_dir = os.path.join(os.getcwd(), config.train_data)
      data = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
      print("[info] Javice: 当前数据集路径{0},数据集规模为 {1} ".format(data_dir,len(data)))
    else:
      data_dir = (os.path.join(os.getcwd(), config.test_data))
      data = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
      print("[info] Javice: 当前数据集路径{0},数据集规模为 {1} ".format(data_dir,len(data)))
    return data

  def imread(self,path,config):
    '''返回读取图片array,并缩放到256的18*13倍'''
    image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image,(config.tw,config.th))
    return image

  def input_setup(self,config):

    if config.is_train:
      data = self.data_list(config)
    else:
      data = self.data_list(config)

    if config.is_train:
      writer = tf.python_io.TFRecordWriter(os.path.join(os.getcwd(),config.train_data_tfrecord))
      j = 0
      for i in xrange(len(data)):
        input_= self.imread(data[i],config)
        if len(input_.shape) == 3:
          h, w, c = input_.shape
          counter = 0
          nx = ny = 0
          for x in range(0, h-config.image_h+1, config.image_h):
            nx += 1 ; ny =0
            for y in range(0, w-config.image_w+1, config.image_w):
              ny += 1
              sub_input = input_[x:x+config.image_h, y:y+config.image_w,0:c]
              sub_input_ = cv2.GaussianBlur(sub_input,(9,9),sigmaX=7) #高斯模糊滤波器

              sub_input = sub_input.reshape([config.image_h, config.image_w, 3]) 
              sub_input_ = sub_input_.reshape([config.image_h, config.image_w, 3])
              counter += 1

              img = sub_input.tostring() #
              label = sub_input_.tostring()

              example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
                    }
                )
              )
              j += 1
              writer.write(example.SerializeToString())
              print("[info] Javice: the image_{0}_{1} has been wrote into {2}...".format(i+1,counter,os.path.join(os.getcwd(),config.train_data_tfrecord)))
          #print("[info] Javice: 原始图片子图片nx = {0},子图片ny = {1},共有 =  {2}：".format(nx,ny,counter)) #[info] Javice: 原始图片子图片nx = 18,子图片ny = 13,共有 =  234：
      print(j)
      writer.close()

    # 解析tfrecord文件

  def parse_tfrecord(self,tfrecord_path,batch_size):
    '''
    return:
        image : Tensor("sub:0", shape=(256, 256, 3), dtype=float32),
        image*1/255 -0.5
        label 为 guass之前的原始HR
    '''
    filename_queue = tf.train.string_input_producer([tfrecord_path],shuffle=False)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features = {
                'label': tf.FixedLenFeature((),tf.string),
                'img': tf.FixedLenFeature((),tf.string),
              }
          )

    image,label = features['img'],features['label']


    decode_img = tf.decode_raw(image,tf.uint8)
    decode_label = tf.decode_raw(label,tf.uint8)

    decode_img = tf.reshape(decode_img,[256,256,3])
    decode_img = tf.reshape(decode_label,[256,256,3])

    decode_img = tf.cast(decode_img,tf.float32)*(1./255) - 0.5
    decode_label = tf.cast(decode_label,tf.float32)*(1./255) - 0.5

    print("[info] : Javice: I have decodeed the data for you ")
    print(decode_img,decode_label)

    return decode_img,decode_label

  #返回 feed_dict参数
  def feed_dict(self,img,label):
    with tf.Session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess,coord=coord)
      for i in range(5):
        img= sess.run([img])
      #img_batch,label_batch = tf.train.batch([img,label],batch_size=batch_size,capacity=256,num_threads=64)
        print(img.shape,type(img))
      coord.request_stop()
      coord.join(threads)
      
    return img,label
      
   # 合成from I-HR' crop 2 complete I-HR
  def merge_image(self,config):
    new_image = []
    crop_image = os.listdir(config.test_data_crop)
    for image in crop_image:
      print("[info] Javice: image",image)
      image = scipy.misc.imread(os.path.join(config.test_data_crop,image)).astype(np.float)
      new_image.append(image)
    print(len(new_image))
    nifb1 = new_image[0]
    nifb2 = new_image[4]
    nifb3 = new_image[8]
    nifb4 = new_image[12]
    nifb5 = new_image[16]
    for i in range(1,4):
      nifb1 = np.concatenate((nifb1,new_image[i]),axis=1)
    for i in range(5,8):
      nifb2 = np.concatenate((nifb2,new_image[i]),axis=1)
    for i in range(9,12):
      nifb3 = np.concatenate((nifb3,new_image[i]),axis=1)
    for i in range(13,16):
      nifb4 = np.concatenate((nifb4,new_image[i]),axis=1)
    for i in range(17,20):
      nifb5 = np.concatenate((nifb5,new_image[i]),axis=1)
    nifb = np.concatenate((nifb1,nifb2),axis=0)
    nifb = np.concatenate((nifb,nifb3),axis=0)
    nifb = np.concatenate((nifb,nifb4),axis=0)
    nifb = np.concatenate((nifb,nifb5),axis=0)
    print(type(nifb),nifb.shape)
    plt.imshow(nifb*1./255)
    plt.grid(True)
    plt.show()
    #self.imsave("E:/Larisvnet/data/testdata_crop_input/new_image_face.jpg",nifb.reshape([3840,3072,3]))

if __name__ == '__main__':
  data = Data()
  config = Config()
 # data.input_setup(config)  # config.is_train : True 生成训练样本tfrecord文件： train-tfrecord
  Ilr,Ihr = data.parse_tfrecord(os.path.join(os.getcwd(),config.train_data_tfrecord),100)

  #data.feed_dict(Ilr,Ihr) "制作训练数据集的问题啊"
