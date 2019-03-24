#！/usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np
import pprint
from data import Data
from model import BasicModule
from model import ResnetGenerator,G,D
from model import Assess
from model import Loss
from model import Train
from model import Config
from utils import Utils

class App():
	'''
	App: to make code easier
	 	class object:
	 		- Data(): 提供数据操作
	 		- BasicModule() : 模型父类，提供训练模型参数的.save/.read
	 		- ResnetGenerator(),G(),D(): Larisv网络的子组件：残差生成网络/顺逆卷积生成网络/卷积判别网络
			- Assess(): Larisv模型评估
			- Loss() : Larisv损失函数
			- Config(): Larisv 训练配置文件
			- Utils(): Project's component项目组件
	'''
	def __init__(self):
		super(App,self).__init__()
		self.data = Data()
		self.bm = BasicModule()
		self.RG = ResnetGenerator()
		self.G = G()
		self.D = D()
		self.assess = Assess()
		self.loss = Loss()
		self.train = Train()
		self.config = Config()
		self.utils = Utils()

	def run(self):
		print("[INFO] Strainger: Hello ,Javice are you there?")
		#Step0: 使用tf.placeholder占位符进行预演
		x = tf.placeholder(tf.float32,shape=[10,837,574,3])
		y = tf.placeholder(tf.float32,shape=[10,837,574,3])
		#Step1: 获取训练集样本
		#data,label = self.data.feed_dict()
		#Step2: 模型训练
		G_1,G_2 = self.G.forward(x),self.G.forward(x)
		S1_D,S2_D,W_D = self.D.forward(G_1),self.D.forward(G_2),self.D.forward(y)
		print("[INFO] Javice: Yes, I have alreadly setup the G_1,G_2,S1_D,S2_D,W_D")
		#hand_input = input("[INFO] Javice: Do you want go on ,or exit,please input [y/n]  ")
		hand_input = 'y'
		if hand_input == "y":
			print("[INFO] Javice: as you wish~",G_1,y)
			mse_loss = tf.Variable(tf.constant([10.]))
			print("[INFO] Javice: I figure out this mse loss is ",mse_loss)
			self.train.train(self.bm,self.config,mse_loss)
		else:
			print("[INFO] Javice: See you~")
		#Step3: larisv 评估
		#Step3: 生成I-SR


if __name__ == '__main__':
	app = App()
	app.run()