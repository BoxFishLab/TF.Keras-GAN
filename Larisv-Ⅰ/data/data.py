#！/usr/bin/env python
#-*- coding: utf-8 -*-

""""""

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

os.environ['TF_CPP_LOG_MIN_LEVEL'] = '1'

try:
  xrange
except:
  xrange = range

class Config():
  def __init__(self):
    '''
    训练 I-LR：(837*574*3)  from 标签I-HR：(4600*3348*3) crop (837*574*3) 按1-32 比例产生子子集
    '''
    self.is_train = True
    self.factor_1 = 3.59375
    self.factor_2 = 3.26953125
    
    self.image_w = 837
    self.image_h = 574

    self.data_root = "E:/Larisv/Larisv-pre/data/"
    self.train_data = "E:/Larisv/Larisv-pre/data/traindata/original-face/"
    self.train_data_crop = "E:/Larisv/Larisv-pre/data/traindata-crop/"
    self.train_label_crop = "E:/Larisv/Larisv-pre/data/trainlabel-crop/"   
    self.test_data = "E:/Larisv/Larisv-pre/data/testdata/"
    self.test_data_crop = "E:/Larisv/Larisv-pre/data/testdata-crop/"
    self.index = 0
   
class Data():
  
  def __init__(self):
    pass

  def prepare_data(self,sess,config):
    """返回图片文件列表"""
    if config.is_train:
      print("config.train_data",config.train_data)  
      filenames = os.listdir(config.train_data)
      data_dir = os.path.join(os.getcwd(), config.train_data)
      data = glob.glob(os.path.join(data_dir, "*.jpg"))
      print("[info] Javice: 当前数据集路径 ",data_dir)
      print("[info] Javice: 返回数据集中的样本列表 {0},数据集规模为 {1}".format(data,len(data)))

    else:
      data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), config.test_data)), "original-face")
      data = glob.glob(os.path.join(data_dir, "*.jpg"))
      print("[info] Javice: 当前数据集路径 ",data_dir)
      print("[info] Javice: 返回数据集中的样本列表 {0},数据集规模为 {1}".format(data,len(data)))
    return data

  def imread(self,path,config):
    '''返回读取图片array'''
    if config.is_train:
      image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
      return image
    
    else:
      image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
      h,w,c = image.shape 
      th = int(config.factor_1*h)
      tw = int(config.factor_2*w)
      image = cv2.resize(image,(tw,th))
      return image

  def input_setup(self,sess,config):

    if config.is_train:
      data = self.prepare_data(sess,config)
    else:
      data = self.prepare_data(sess,config)

    if config.is_train:
      for i in xrange(len(data)):
        input_= self.imread(data[i],config)
        if len(input_.shape) == 3: #print(len(input_.shape) == 3) True
          h, w, c = input_.shape
         # print("[info] Javice: the image.shape is ",input_.shape)
          o_counter = 0
          nx = ny = 0
          for x in range(0, h-config.image_h+1, config.image_h):
            nx += 1 ; ny =0
            for y in range(0, w-config.image_w+1, config.image_w):
              ny += 1
             # print("[info] Javice: x = {0}, y= {1}".format(x,y))
              sub_input = input_[x:x+config.image_h, y:y+config.image_w,0:c]
              sub_input_ = cv2.GaussianBlur(sub_input,(7,7),sigmaX=5.5)
              sub_input = sub_input.reshape([config.image_h, config.image_w, 3]) 
              sub_input_ = sub_input_.reshape([config.image_h, config.image_w, 3])
              o_counter += 1
              self.imsave(sub_input_,config.train_data_crop+"{0}_{1}.jpg".format(i+1,o_counter))
              self.imsave(sub_input,config.train_label_crop+"{0}_{1}.jpg".format(i+1,o_counter))
              print("[info] Javice: config.train_data_crop path: ",config.train_data_crop+"{0}_{1}.jpg".format(i+1,o_counter))
         # print("[info] Javice: 原始图片子图片nx = {0},子图片ny = {1},共有 =  {2}：".format(nx,ny,o_counter))
    else:
      input_ = self.imread(data[config.index],config)
      if len(input_.shape) == 3: #print(len(input_.shape) == 3) True
        h, w, c = input_.shape
      counter = 0
      nx = ny = 0 
      for x in range(0, h-config.image_h+1, config.image_h):
        nx += 1; ny = 0
        for y in range(0, w-config.image_w+1, config.image_w):
          ny += 1
          sub_input = input_[x:x+config.image_h, y:y+config.image_w,0:3]
          sub_input = sub_input.reshape([config.image_h, config.image_w, 3])  
          counter += 1
          self.imsave(sub_input,config.test_data_crop+"{0}.jpg".format(counter))
          print("[info] Javice: config.test_data_crop: ",config.test_data_crop+"{0}.jpg".format(counter))
      print("[info] Javice: 原始图片子图片nx = {0},子图片ny = {1},共有 =  {2}张子图： {3}".format(nx,ny,counter,nx*ny == counter))

    def imsave(self,image, path):
      return cv2.imwrite(path,image,[int(cv2.IMWRITE_JPEG_QUALITY),100])

#     if not config.is_train:
#       return nx, ny

    # 制成tfrecord文件    
    def save_to_tfrecord(self,cd_list,label,tfrecord_path):
        '''
        index 0 cats D:/Charben/_Datasets/dogcat/train_data/cats/
        index 1 dogs D:/Charben/_Datasets/dogcat/train_data/dogs/
        image size : 64,64,3
        '''
        writer = tf.python_io.TFRecordWriter(tfrecord_path+"train.tfrecord")
        for x,y in zip(cd_list,label):
            print(x,y)
            image = Image.open(x)
            # half_the_width = image.size[0]/2
            # half_the_height = image.size[1]/2
            # image = image.crop(
            #     ( 
            #         half_the_width - 112,
            #         half_the_height - 112,
            #         half_the_width + 112,
            #         half_the_height + 112,
            #     )
            # )
            image = image.resize((64,64))
            img_raw = image.tobytes() #转为二进制
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                )
            )
            writer.write(example.SerializeToString())
        print("[info]: 数据转化完成")
        writer.close()

    # 解析tfrecord文件
    def parse_tfrecord(self,tfrecord_path,batch_size):
        '''
        return:
            image : Tensor("sub:0", shape=(224, 224, 3), dtype=float32),
            image*1/255 -0.5
            class：Tensor("Cast_1:0", shape=(), dtype=int32)
        '''
        filename_queue = tf.train.string_input_producer([tfrecord_path],shuffle=False)
        reader = tf.TFRecordReader()
        _,serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
                serialized_example,
                features = {
                    'label': tf.FixedLenFeature([],tf.int64),
                    'img_raw': tf.FixedLenFeature([],tf.string),
                }
            )
        image,label = features['img_raw'],features['label']
        decode_img = tf.decode_raw(image,tf.uint8)
        decode_img = tf.reshape(decode_img,[224,224,3])
        decode_img = tf.cast(decode_img,tf.float32)*(1./255) - 0.5
        label = tf.cast(label,tf.int32)
        # print("[info] : 解析后数据：{0},类标：{1}".format(decode_img,label))
        return decode_img,label

    #返回 feed_dict参数
    def feed_dict(self,tfrecord_path,batch_size):
        img,label = self.parse_data(tfrecord_path,batch_size)
        img_batch,label_batch = tf.train.batch([img,label],batch_size=batch_size,capacity=256,num_threads=64)
        #print(label_batch,type(label_batch))
        #Tensor("shuffle_batch:1", shape=(100,), dtype=int32)
        return img_batch,label_batch

      
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
  with tf.Session() as sess:
      data.input_setup(sess,config)
#  data.merge_image(config)
