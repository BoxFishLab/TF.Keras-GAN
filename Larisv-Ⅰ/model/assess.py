#!/usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf

class Assess(object):
	'''
	性能评估指标
	'''
	def __init__(self):
		pass
	def accuracy(self,y,y_):
		with tf.name_scope('accuracy'):   
			with tf.name_scope('correct_prediction'):
				correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
			with tf.name_scope('accuracy'):
				accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
			tf.summary.scalar('accuracy',accuracy)
		return accuracy
		
	def testPSNR():
		avg_psnr = 0
		for batch in testing_data_loader:
			input, target = Variable(batch[0]), Variable(batch[1])
			if opt.cuda:
				input = input.cuda()
				target = target.cuda()
			wp = G_1(input)
			prediction = G_2(wp)
			mse = loss_MSE(prediction, target)
			psnr = 10 * log10(1 / mse.data[0])
			avg_psnr += psnr
			ret = avg_psnr / len(testing_data_loader)
			print("----> Avg. PSNR: {:.4f} dB".format(ret))
		return ret