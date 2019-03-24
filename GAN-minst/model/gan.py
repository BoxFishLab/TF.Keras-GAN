#!/usr/bin/env python
#-*- coding: utf-8- -*-

import tensorflow as tf
import ops

class D(object):

	def __init__(self):
		super(D,self).__init__()
		self.pgan = PGANnet()    #创建pgan对象方便重用部分函数
		self.weights ={
			'Dw1': self.pgan.init_weights([4,4,3,32],name='Dw1'),
			'Dw2': self.pgan.init_weights([4,4,32,64],name='Dw2'),
			'Dw3': self.pgan.init_weights([4,4,64,128],name='Dw3'),
			'Dw4': self.pgan.init_weights([4,4,128,256],name='Dw4'),
			'Dw5': self.pgan.init_weights([4,4,256,1],name='Dw5'),
		}
		self.biases = {
			'Db1': self.pgan.init_bias([32],'Db1'),
			'Db2': self.pgan.init_bias([64],'Db2'),
			'Db3': self.pgan.init_bias([128],'Db3'),
			'Db4': self.pgan.init_bias([256],'Db4'),
			'Db5': self.pgan.init_bias([1],'Db5'),
		}

	def forward(self,input_x):

		
		conv1 = self.pgan.conv2d(input_x,self.weights['Dw1'],self.biases['Db1'],strides=[1,2,2,1])
		conv1_af_relu = self.pgan.af_leaky_relu(conv1,0.2)
		conv1_batch_norm2d = self.pgan.batch_norm2d(conv1_af_relu)

		conv2 = self.pgan.conv2d(conv1_batch_norm2d,self.weights['Dw2'],self.biases['Db2'],strides=[1,2,2,1])
		conv2_af_relu = self.pgan.af_leaky_relu(conv2,0.2)
		conv2_batch_norm2d = self.pgan.batch_norm2d(conv2_af_relu)

		conv3 = self.pgan.conv2d(conv2_batch_norm2d,self.weights['Dw3'],self.biases['Db3'],strides=[1,2,2,1])
		conv3_af_relu = self.pgan.af_leaky_relu(conv3,0.2)
		conv3_batch_norm2d = self.pgan.batch_norm2d(conv3_af_relu)

		conv4 = self.pgan.conv2d(conv3_batch_norm2d,self.weights['Dw4'],self.biases['Db4'],strides=[1,2,2,1])
		conv4_af_relu = self.pgan.af_leaky_relu(conv4,0.2)
		conv4_batch_norm2d = self.pgan.batch_norm2d(conv4_af_relu)

		conv5 = self.pgan.conv2d(conv4_batch_norm2d,self.weights['Dw5'],self.biases['Db5'],strides=[1,2,2,1])
		conv5_af_relu = self.pgan.af_leaky_relu(conv5,0.2)
		conv5_batch_norm2d = self.pgan.batch_norm2d(conv5_af_relu)
		output = self.pgan.af_sigmoid(conv5_batch_norm2d)

		# print("[info] Javice: DisNet")
		# print("		-----: input_x.shape : ",input_x.shape)
		# print("		-----: conv2d_layer1.shape: ",conv1_batch_norm2d.shape)
		# print("		-----: conv2d_layer2.shape: ",conv1_batch_norm2d.shape)
		# print("		-----: conv2d_layer3.shape: ",conv1_batch_norm2d.shape)
		# print("		-----: conv2d_layer4.shape: ",conv4_batch_norm2d.shape)
		# print("		-----: conv2d_layer5.shape: ",conv5_batch_norm2d.shape)
		# print("		-----: conv2d_output.shape: ",output.shape)

		# 是否要求最后需要一层全连接网络
		# output_shape = net_17.get_shape().as_list()
		# nodes = output_shape[1]*output_shape[2]*output_shape[3]
		# output = tf.reshape(output,[-1,nodes])
		# print("		-----: output.shape: ",output.shape)

		return output

# Generator
class G():

	def __init__(self):
		super(G, self).__init__()
		self.pgan = PGANnet()
		
		self.weights = {
			'Gw1': self.pgan.init_weights([4,4,3,32],name='Gw1'),
			'Gw2': self.pgan.init_weights([4,4,32,64],name='Gw2'),
			'Gw3': self.pgan.init_weights([4,4,64,128],name='Gw3'),
			'Gw4': self.pgan.init_weights([4,4,128,256],name='Gw4'),
			'Gw5': self.pgan.init_weights([4,4,256,512],name='Gw5'),
			'Gw6': self.pgan.init_weights([4,4,512,1024],name='Gw6'),
			'Gw7': self.pgan.init_weights([4,4,1024,512],name='Gw7'),
			'Gw8': self.pgan.init_weights([1,1,512,512],name='Gw8'),
		}

		self.biases = {
			'Gb1': self.pgan.init_bias([32],'Gb1'),
			'Gb2': self.pgan.init_bias([64],'Gb2'),
			'Gb3': self.pgan.init_bias([128],'Gb3'),
			'Gb4': self.pgan.init_bias([256],'Gb4'),
			'Gb5': self.pgan.init_bias([512],'Gb5'),
			'Gb6': self.pgan.init_bias([1024],'Gb6'),
			'Gb7': self.pgan.init_bias([512],'Gb7'),
			'Gb8': self.pgan.init_bias([512],'Gb8'),
		}
		
		self.dweights ={
			'dGw1': self.pgan.init_weights([4,4,512,512],name='dGw1'),
			'dGw2': self.pgan.init_weights([4,4,1024,512],name='dGw2'),
			'dGw3': self.pgan.init_weights([4,4,512,1024],name='dGw3'),
			'dGw4': self.pgan.init_weights([4,4,256,512],name='dGw4'),
			'dGw5': self.pgan.init_weights([4,4,128,256],name='dGw5'),
			'dGw6': self.pgan.init_weights([4,4,64,128],name='dGw6'),
			'dGw7': self.pgan.init_weights([4,4,32,64],name='dGw7'),
			'dGw8': self.pgan.init_weights([4,4,3,32],name='dGw8'),
		}

	def forward(self,input_x):

		conv1 = self.pgan.conv2d(input_x,self.weights['Gw1'],self.biases['Gb1'],strides=[1,2,2,1])
		conv1_af_relu = self.pgan.af_leaky_relu(conv1,0.2)
		conv1_batch_norm2d = self.pgan.batch_norm2d(conv1_af_relu)
		
		conv2 = self.pgan.conv2d(conv1_batch_norm2d,self.weights['Gw2'],self.biases['Gb2'],strides=[1,2,2,1])
		conv2_af_relu = self.pgan.af_leaky_relu(conv2,0.2)
		conv2_batch_norm2d = self.pgan.batch_norm2d(conv2_af_relu)

		conv3 = self.pgan.conv2d(conv2_batch_norm2d,self.weights['Gw3'],self.biases['Gb3'],strides=[1,2,2,1])
		conv3_af_relu = self.pgan.af_leaky_relu(conv3,0.2)
		conv3_batch_norm2d = self.pgan.batch_norm2d(conv3_af_relu)

		conv4 = self.pgan.conv2d(conv3_batch_norm2d,self.weights['Gw4'],self.biases['Gb4'],strides=[1,2,2,1])
		conv4_af_relu = self.pgan.af_leaky_relu(conv4,0.2)
		conv4_batch_norm2d = self.pgan.batch_norm2d(conv4_af_relu)

		conv5 = self.pgan.conv2d(conv4_batch_norm2d,self.weights['Gw5'],self.biases['Gb5'],strides=[1,2,2,1])
		conv5_af_relu = self.pgan.af_leaky_relu(conv5,0.2)
		conv5_batch_norm2d = self.pgan.batch_norm2d(conv5_af_relu)

		'''
		conv5_batch_norm2d = ResnetGenerator(conv5_batch_norm2d).forward(conv5_batch_norm2d)
		'''
		conv6 = self.pgan.conv2d(conv5_batch_norm2d,self.weights['Gw6'],self.biases['Gb6'],strides=[1,2,2,1])
		conv6_af_relu = self.pgan.af_leaky_relu(conv6,0.2)
		conv6_batch_norm2d = self.pgan.batch_norm2d(conv6_af_relu)

		conv7 = self.pgan.conv2d(conv6_batch_norm2d,self.weights['Gw7'],self.biases['Gb7'],strides=[1,2,2,1])
		conv7_af_relu = self.pgan.af_leaky_relu(conv7,0.2)
		conv7_batch_norm2d = self.pgan.batch_norm2d(conv7_af_relu)
		
		conv8 = self.pgan.conv2d(conv7_batch_norm2d,self.weights['Gw8'],self.biases['Gb8'],strides=[1,2,2,1])
		conv8_af_relu = self.pgan.af_leaky_relu(conv8,0.2)

		# print("[info] Javice: GeneratorNet")
		# print("		-----: input_x.shape : ",input_x.shape)
		# print("		-----: conv2d_layer1.shape: ",conv1_batch_norm2d.shape)
		# print("		-----: conv2d_layer2.shape: ",conv2_batch_norm2d.shape)
		# print("		-----: conv2d_layer3.shape: ",conv3_batch_norm2d.shape)
		# print("		-----: conv2d_layer4.shape: ",conv4_batch_norm2d.shape)
		# print("		-----: conv2d_layer5.shape: ",conv5_batch_norm2d.shape)
		# print("		-----: conv2d_layer6.shape: ",conv6_batch_norm2d.shape)
		# print("		-----: conv2d_layer7.shape: ",conv7_af_relu.shape)
		# print("		-----: conv2d_layer8.shape: ",conv8_af_relu.shape)

		dconv1 = self.pgan.dconv2d(self.pgan.af_relu(conv8),self.dweights['dGw1'],[10,7,5,512],[1,2,2,1])
		dconv1_batch_norm2d = self.pgan.batch_norm2d(dconv1)
		dconv1_dropout = self.pgan.drop_out(dconv1_batch_norm2d,0.5)
		dconv1_conv7 = tf.concat([dconv1_dropout,conv7],1)
		
		dconv2 = self.pgan.dconv2d(self.pgan.af_relu(dconv1_conv7),self.dweights['dGw2'],[10,14,9,1024],[1,2,2,1])
		dconv2_batch_norm2d = self.pgan.batch_norm2d(dconv2)
		dconv2_dropout = self.pgan.drop_out(dconv2_batch_norm2d,0.5)
		dconv2_conv6 = tf.concat([dconv2_dropout,conv6],1)

		dconv3 = self.pgan.dconv2d(self.pgan.af_relu(dconv2_conv6),self.dweights['dGw3'],[10,27,18,512],[1,2,2,1])
		dconv3_batch_norm2d = self.pgan.batch_norm2d(dconv3)
		dconv3_dropout = self.pgan.drop_out(dconv3_batch_norm2d,0.5)
		dconv3_conv5 = tf.concat([dconv3_dropout,conv5],1)

		dconv4 = self.pgan.dconv2d(self.pgan.af_relu(dconv3_conv5),self.dweights['dGw4'],[10,53,36,256],[1,2,2,1])
		dconv4_batch_norm2d = self.pgan.batch_norm2d(dconv4)
		dconv4_conv4 = tf.concat([dconv4_batch_norm2d,conv4],1)

		dconv5 = self.pgan.dconv2d(self.pgan.af_relu(dconv4_conv4),self.dweights['dGw5'],[10,105,72,128],[1,2,2,1])
		dconv5_batch_norm2d = self.pgan.batch_norm2d(dconv5)
		dconv5_conv3 = tf.concat([dconv5_batch_norm2d,conv3],1)

		dconv6 = self.pgan.dconv2d(self.pgan.af_relu(dconv5_conv3),self.dweights['dGw6'],[10,210,144,64],[1,2,2,1])
		dconv6_batch_norm2d = self.pgan.batch_norm2d(dconv6)
		dconv6_conv2 = tf.concat([dconv6_batch_norm2d,conv2],1)

		dconv7 = self.pgan.dconv2d(self.pgan.af_relu(dconv6_conv2),self.dweights['dGw7'],[10,419,287,32],[1,2,2,1])
		dconv7_batch_norm2d = self.pgan.batch_norm2d(dconv7)
		dconv7_conv1 = tf.concat([dconv7_batch_norm2d,conv1],1)

		dconv8 = self.pgan.dconv2d(self.pgan.af_relu(dconv7_conv1),self.dweights['dGw8'],[10,837,574,3],[1,2,2,1])
		output = self.pgan.af_tanh(dconv8)

		# print("------"*10)
		# print("		-----: dconv2d_layer1.shape: ",dconv1_dropout.shape,"dconv1_conv7_catlayer1.shape: ",dconv1_conv7.shape)
		# print("		-----: dconv2d_layer2.shape: ",dconv2_dropout.shape,"dconv2_conv6_catlayer2.shape: ",dconv2_conv6.shape)
		# print("		-----: dconv2d_layer3.shape: ",dconv3_dropout.shape,"dconv3_conv5_catlayer3.shape: ",dconv3_conv5.shape)
		# print("		-----: dconv2d_layer4.shape: ",dconv4_batch_norm2d.shape,"dconv4_conv4_catlayer4.shape",dconv4_conv4.shape)
		# print("		-----: dconv2d_layer5.shape: ",dconv5_batch_norm2d.shape,"dconv5_conv3_catlayer5.shape",dconv4_conv4.shape)
		# print("		-----: dconv2d_layer6.shape: ",dconv6_batch_norm2d.shape,"dconv6_conv2_catlayer6.shape",dconv6_conv2.shape)
		# print("		-----: dconv2d_layer7.shape: ",dconv7_batch_norm2d.shape,"dconv7_conv1_catlayer7.shape",dconv7_conv1.shape)
		# print("		-----: dconv2d_output.shape: ",dconv8.shape)
		return output

#残差生成器
class ResnetGenerator():
	def __init__(self):
		super(ResnetGenerator, self).__init__()
		self.pgan = PGANnet()
		self.resblock = ResBlock()

		# self.RGweights = {
		# 	'RGw1': self.pgan.init_weights([4,4,3,32],name='RGw1'),
		# }

		# self.RGbiases = {
		# 	'RGb1': self.pgan.init_bias([32],'RGb1'),
		# }

	def forward(self,input_x,res_block=3):

		# input_x = self.pgan.af_relu(self.pgan.batch_norm2d(self.pgan.conv2d(
		# 	input_x,
		# 	self.RGweights['RGw1'],
		# 	self.RGbiases['RGb1'],
		# 	strides=[1,2,2,1]
		# 	)))
		for i in range(res_block):
			input_x = self.resblock.res_block(input_x)

		return input_x

class ResBlock():

	def __init__(self):
		super(ResBlock,self).__init__()
		self.pgan = PGANnet()

	def res_block(self,input_x):

		_,_,_,self.in_channels = input_x.get_shape().as_list()

		self.Rweights = {
			'Resw1': self.pgan.init_weights([1,1,self.in_channels,64],name='Resw1'),
			'Resw2': self.pgan.init_weights([3,3,64,512],name='Resw2'),
			'Resw3': self.pgan.init_weights([1,1,512,3],name='Resw3'),
		}
		self.Rbiases = {
			'Resb1': self.pgan.init_bias([64],'Resb1'),
			'Resb2': self.pgan.init_bias([512],'Resb2'),
			'Resb3': self.pgan.init_bias([3],'Resb3'),
		}
		net = self.pgan.batch_norm2d(self.pgan.conv2d(input_x,self.Rweights['Resw1'],self.Rbiases['Resb1'],strides=[1,1,1,1]))
		net = self.pgan.conv2d(net,self.Rweights['Resw2'],self.Rbiases['Resb2'],strides=[1,1,1,1])
		net = self.pgan.conv2d(net,self.Rweights['Resw3'],self.Rbiases['Resb3'],strides=[1,1,1,1])
		return input_x+net



