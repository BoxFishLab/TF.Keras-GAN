#！/usr/bin/env python
#-*- coding: utf-8 -*-

class Config(object):

	batch_size = 10  # 100计算机停止工作了
	keep_prob = 0.5
	LEARNING_BASE_RATE = 0.1
	LEARNING_RATE_SPEED = 100
	LEARNING_DECAY_RATE = 0.95
	STAIRCASE = True
	TRAINING_STEPS = 100
	tfrecord_path = "D:/Charben/_Pandora_tensorflow/CatDogs/data/tfrecord/"
	model_vali_path = "D:/Charben/_Pandora_tensorflow/CatDogs/_cahe/-0.meta"
	model_data_path = "D:/Charben/_Pandora_tensorflow/CatDogs/_cahe/-0"
	model_param_path = "D:/Charben/_Pandora_tensorflow/CatDogs/_cahe/"
	log_path = "D:/Charben/_Pandora_tensorflow/CatDogs/_log/"
	
	# ### Load data
	# print('----> Loading data......')
	# root_path = "dataset/"
	# train_set = getTrainData(root_path + opt.dataset)
	# test_set = getTestData(root_path + opt.dataset)
	# training_data_loader = DataLoader(dataset = train_set, num_workers = opt.threads, batch_size = batch_size, shuffle = True)
	# testing_data_loader = DataLoader(dataset = test_set, num_workers = opt.threads, batch_size = batch_size, shuffle = False)


	def fix(self,kwargs):
		'''
		更改配置文件值
		'''
		for k,v in kwargs.items():
			if hasattr(k,self):
				setattr(k,v)
			else:
				print("[information]:plaese check your config set")
