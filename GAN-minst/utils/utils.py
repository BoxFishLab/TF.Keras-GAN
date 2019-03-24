#!/usr/bin/env python 
# -*- coding: utf-8 -*-

'''
class utils:
	Variables:
		feature: 特征图
	functions:
		computer_graph: 生成计算图
		vis_feature(): 可视化特征图
		feature_graph():
'''
class Utils(object):

	def __init__(self):
		super(Utils,self)
		pass

	def log(self,config_log_path):
		'''
		输出日志文件：用于可视化
		'''
		writer = tf.summary.FileWriter(config_log_path,tf.get_default_graph())
		print("完成了日志写入！")
		writer.close()
		
	def vis_feature(self,config,image):
		classes = ['cat','dog']		
		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver = tf.train.import_meta_graph(config.model_vali_path)
			print("成功导入模型meta")
			saver.restore(sess,config.model_data_path)
			print("成功导入模型data")
			image = image.eval()
			graph = tf.get_default_graph()
			x_ = graph.get_operation_by_name('input_x').outputs[0]
			conv_relu1 = graph.get_collection('conv_relu1')[0]
			conv_relu2 = graph.get_collection('conv_relu2')[0]
			conv_relu3 = graph.get_collection('conv_relu3')[0]
			conv_relu4 = graph.get_collection('conv_relu4')[0]
			pool_relu5 = graph.get_collection('pool_relu5')[0]
			conv_relu17 = graph.get_collection('conv_relu17')[0]
			y,conv_relu1,conv_relu2,conv_relu3,conv_relu4,pool_relu5,conv_relu17 = sess.run([y,conv_relu1,conv_relu2,conv_relu3,conv_relu4,pool_relu5,conv_relu17],feed_dict={x_:image})
			'''选择特征层可视化'''
			self.feature_graph(conv_relu17)
	def feature_graph(self,feature_graph):
		feature_graph_combination = []
		plt.figure()
		for i in range(feature_graph.shape[0]):
			simgle_feature_graph = feature_graph[i,:,:,:]
			print(simgle_feature_graph.shape)
			for j in range(feature_graph.shape[-1]):
				sig = simgle_feature_graph[:,:,j]
				feature_graph_combination.append(sig)
		feature_graph_sum = np.sum(ele for ele in feature_graph_combination)
		plt.imshow(feature_graph_sum)
		plt.savefig('feature_graph_relu_17.png')
		plt.show()

