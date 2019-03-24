#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
Loss多元感知损失
	content loss:

loss_BCE = nn.BCELoss()
loss_MSE = nn.MSELoss()
loss_L1 = nn.L1Loss()

'''
import tensorflow as tf
class Loss():
	'''
	P-GAN:Loss functions design
		content loss:
		gan loss:
	'''
	def __init__(self):
		pass

	def loss(self,y_,y):
		with tf.name_scope("loss_function"):
			loss = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
			tf.summary.scalar('loss',loss)
		return loss
	
	def mse_loss(self,G,y):
		mse_loss = 0 
		batch_size,height,width,channel = G.shape
		for i in range(batch_size):
			G_ = G[i,:,:,:]
			y_ = y[i,:,:,:]
			for j in range(channel):
				G_s = G_[:,:,j]
				y_s = y_[:,:,j]
			mse_loss += tf.reduce_mean(G-y)
		return mse_loss

	def style_loss(self,target_feature,style_feature):
		_,height,width,channel = map(lambda i:i.value,target_feature.get_shape())
		target_size = height*width*channel
		target_feature = tf.reshape(target_feature,(-1,channel))
		target_gram = tf.matmul(tf.transpose(target_feature),target_feature)/target_size
		style_feature = tf.reshape(style_feature,(-1,channel))
		style_gram = tf.matmul(tf.transpose(style_feature),target_feature)/target_size
		return tf.nn.l2_loss(target_gram-style_gram)/target_size
	
	def loss_function(self,content_image,style_image,target_image):
		style_feature = self.style_graph
		content_feature = self.content_graph
		target_feature = self.vgg19([target_image])
		loss = 0.0
		for layer in self.CONTENT_LAYERS:
			loss += self.CONTENT_WEIGHT*self.content_loss(target_feature[layer],content_feature[layer])
		for layer in self.STYLE_LAYERS:
			loss += self.STYLE_WEIGHT*self.style_loss(target_feature[layer],style_feature[layer])
		return loss	
