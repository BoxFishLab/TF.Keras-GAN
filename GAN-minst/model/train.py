#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os 

'''
Class Train(BasicModule):
	继承基础模块
'''

class Train():

	def __init__(self):
		super(Train,self).__init__()
		pass

	def optimizer(self,config,loss,step):
		global_step = step
		learning_rate = tf.train.exponential_decay(
			config.LEARNING_BASE_RATE,
			global_step,
			config.LEARNING_RATE_SPEED,
			config.LEARNING_DECAY_RATE,
			config.STAIRCASE
		)
		train_step_G = tf.train.GradientDescentOptimizer(learning_rate)
		print("Are you all right")
		train_step_G = train_step_G.minimize(loss,global_step=global_step)
		train_step_S = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
		train_step_W = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
		return train_step_G,train_step_S,train_step_W

	def train(self,model,config,loss):
		step = tf.Variable(0)
		G_,S_,W_ = self.optimizer(config,loss,step)
		merged = tf.summary.merge_all()
		with tf.Session() as sess:
			# summary_weiter = tf.summary.FileWriter(config.log_path,sess.graph)
			init_op = tf.global_variables_initializer() #初始化所有变量
			sess.run(init_op)
			# coord = tf.train.Coordinator()
			# threads = tf.train.start_queue_runners(sess=sess,coord=coord)
			print("正在训练中....请等待")
			# for step in range(config.TRAINING_STEPS):
			# 	for i in range(0,data_size,config.batch_size):
			# 		img_batch = np.array(image[i:i+config.batch_size])
			# 		label_batch = np.array(label[i:i+config.batch_size]).reshape(self.config.batch_size,1)
			# 		feed_dict={x_:img_batch,y_:label_batch}
			# 		summary,_ = sess.run([merged,train_step],feed_dict=feed_dict)
			# 		summary_weiter.add_summary(summary,i)
			# 		print("After trianing {0}/{1} times,loss: {2},the accuracy: {3}".format(step,i,sess.run(loss,feed_dict=feed_dict),sess.run(accuracy,feed_dict=feed_dict)))
			# 	if step % 10 == 0:
			# 		print("After trianing {0} times,loss: {1},the accuracy: {2}".format(step,sess.run(loss,feed_dict=feed_dict),sess.run(accuracy,feed_dict=feed_dict)))
			# 		model.save(config.model_param_path,global_step=step)					
			# coord.request_stop()
			# coord.join(threads)
			# summary_weiter.close()
			graph = tf.get_default_graph()
			write = tf.summary.FileWriter(os.path.join(os.getcwd(),"/tfgraph.graph"),graph)
			print("完成Larisv图的设计")
			write.close()