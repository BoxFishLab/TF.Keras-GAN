#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
Easier to build model based on tensorflow
'''
import tensorflow as tf

def conv2d(self,input_x,kernel,strides):
	with tf.name_scope("conv2d"):
		conv2d = tf.nn.conv2d(
			input_x,
			kernel,
			strides=strides,
			padding="SAME"
		)
		return conv2d

def conv2d_add_bias(self,input_x,kernel,bias,strides):
	with tf.name_scope("conv2d"):
		conv2d = tf.nn.conv2d(
			input_x,
			kernel,
			strides=strides,
			padding="SAME"
		)
		conv2d_add_bias = tf.nn.bias_add(conv2d,bias)
		return conv2d_add_bias

def dconv2d(self,input_x,n_filter,output_shape,strides):
	'''反卷机操作
	'''
	return tf.nn.conv2d_transpose(input_x,n_filter,output_shape,strides=strides)

def init_weights(self,shape,name):
	return tf.Variable(tf.random_normal(shape,stddev=1e-3,name=name))

def init_bias(self,shape,name):
	return tf.Variable(tf.zeros(shape,name=name))
	
# 激活函数 : activate_func as af	
def lrelu(x,a):
	x = tf.identity(x)
	return (0.5*(1+a))*x + (0.5*(1-a))*tf.abs(x)

def relu(input_x):
	return tf.nn.relu(input_x)
	
def tanh(input_x):
	return tf.nn.tanh(input_x)

def sigmoid(input_x):
	return tf.nn.sigmoid(input_x)

def leaky_relu(self,input_x,apha):
	return tf.where(input_x>0,input_x,input_x*apha)

# def af_leaky_relu(self,input_x,apha):
# 	return tf.nn.leaky_relu(input_x,apha)
'''
由于tensorflow版本原因：
	AttributeError: module 'tensorflow.python.ops.nn' has no attribute 'leaky_relu'
	下：tf.where实现
'''

# 加快模型训练与防止过拟合
def batch_norm2d(self,input_x):
	mean,var = tf.nn.moments(input_x,axes=[0,1,2])
	return tf.nn.batch_normalization(input_x,mean,var,0,1,1e-5)

def drop_out(self,input_x,prob):
	return tf.nn.dropout(input_x,prob)

