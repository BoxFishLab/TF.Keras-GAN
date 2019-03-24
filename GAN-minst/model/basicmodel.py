#！/usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf

class BasicModule():
	def save(self,train_file,global_step):
		#train_file = "D:/Charben/_Pandora.tensorflow版/.cache/"+"train_" + str(self.__class__.__name__)
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver.save(sess,train_file,global_step=global_step)
			print("第",global_step,"次，模型已经保存！")
	def load(self,model_path):
		ckpt = tf.train.get_checkpoint_state(model_path)
		print(type(ckpt))
		return ckpt	
