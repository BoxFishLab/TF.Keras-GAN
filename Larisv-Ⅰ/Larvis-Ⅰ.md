### Larvis-Ⅰ


> IDE：
	python: 3.6.7
	tensorflow-gpu: 1.3.0
	numpy: 1.16.2
	matploylib: 3.0.3
	scipy: 1.2.1
	pillow: 5.4.1
	cuda: 8.0
	cudnn: 6.0
	c++ redis: 2017
	#subtext: 3.3.1
	Eclipse
	jdk 
	pydev插件

> GPU infomation:
	name: GeForce GTX 1080 Ti
	major: 6 minor: 1 memoryClockRate (GHz) 1.6325
	pciBusID 0000:01:00.0
	Total memory: 11.00GiB
	Free memory: 9.09GiB

> CPU infomation:
	处理器: intel(R) Core(TM) i5
	内存: 8G

### 工作文档笔记

`Step pre`:
	
	样本标记(2019/3/6)：
		样本ID：101-200 类标：{黑色素：1，荧光:2} 生成文件：样本ID.json
		问题标记样本ID：{None}
		错误标记样本ID：{None}
		样本总量： 100个;标记时速：18张/h;问题率：None%;出错率：None%；
	已打包(1-200)发送，发送人：柳伟，发送时间：2019/3/6 17:23
	样本标记(2019/3/7)：
		样本ID：301-400 类标：{黑色素：1，荧光:2} 生成文件：样本ID.json
		问题标记样本ID：{ID351：问题图(区域编码错误)}
		错误标记样本ID：{None}
		样本总量： 100个;标记时速：22张/h;问题率：1%;出错率：None%；
	已打包(201-400)发送，发送人：柳伟，发送时间：2019/3/7 16:44
	样本标记(2019/3/8)：
		样本ID：501-600 类标：{黑色素：1，荧光:2} 生成文件：样本ID.json
		问题标记样本ID：{None}
		错误标记样本ID：{None}
		样本总量： 100个;标记时速：23张/h;问题率：None%;出错率：None%；
	将打包(401-600)发送，发送人：柳伟 发送时间：2019/3/8 15:45

	样本标记(2019/3/11)：
		样本ID：701-800 类标：{黑色素：1，荧光:2} 生成文件：样本ID.json
		问题标记样本ID：{None}
		错误标记样本ID：{None}
		样本总量： 100个;标记时速：张/h;问题率：None%;出错率：None%；
	将打包(601-800)发送，发送人：柳伟 发送时间：2019/3/11 15:45

`Step1`: Has Already solved
	

	增强图片信息：
		1. color(色彩): 
		2. brightsness(亮度)
		3. sharpness(锐化)
		4. contract(对比度)
Bed 样本1：
	![image]("E:/Larisvnet/data/sample/1/uvg.jpg")
	图片样本：
		1. 过于黑暗，单方面提高图像亮度、对比度会造成色彩偏失去
		2. 采用通道均衡，不同图片的通道不同，难以调衡
Bed 样本2：
	图片样本：
		1. 过于曝光，单方面提高降低亮度、对比度也会造成色彩偏失去
	![image]("E:/Larisvnet/data/sample/1/uvg.jpg")
good 样本：
	图片样本：
		1. 图片亮度适中，色差适中
	![image]("E:/Larisvnet/data/sample/298/uvg.jpg")

解决Bed 样本的可行性方案：
	a. 
		人工区分图片的明暗度，将图片分类为：
			drak/
			light/
		对dark/目录下的暗图片进行亮度提升
		对light/目录下的高亮图片进行暗度降低
		a.1 采用图片直接增强：
			image.enhance(color,brightness,contract,sharpness)
		优缺点：
			优： 简便易实行
			缺： 1.人力成本随着样本增多而增大
		a.2 采用图像直方图均衡
			r,g,b = split(image)
			for hist in r/g/b:
				'比较各通道的均衡值'
				then:
					d对较大均衡影响的进行直方图均衡处理
			image = image.merge(rf,gf,bf) #将通道图像合成
		优缺点：
			优： 简便易实行，有效避免人力成本
			缺： 1.图像增强不明显 2.图像通道上容易产生偏色现象
		a.3 采用深度学习训练自动增强图片
			训练数据集对：（dark_image,light_image）
			模型选择： SRGAN生成模型
			尽可能生成good样本
		优缺点：
			优： 图像随着训练迭代加深能到达到目标样本
			缺： 1.训练集要求大，训练周期长

`Step1_编译细节`:
	
	安卓环境配置
		1. SDK
		2. ndk环境搭建


`Step2`: Local Region P-GAN for image resolution 图像超分辨重建

	任务描述
		"""
		从低分辨率(image_x)学习到高分辨率的映射F(image_x)的过程可以近似的理解为图像超分辨重建
		修正SRDCNN模型使之能够完成高像素样本训练策略：
		"""

	
	Step2.1 设计网络架构模型
		Step2.1 网络总体架构
			<b>已完成·</b>
		Step2.2 组件网络
			Step2.2.1 判别网络
				a. 网络深度
				b. 网络参数信息: [tf.Tensor,dtype=float32]
				[info] Javice: DisNet
					-----: input_x.shape :  (10, 837, 574, 3)
					-----: conv2d_layer1.shape:  (10, 419, 287, 32)
					-----: conv2d_layer2.shape:  (10, 419, 287, 32)
					-----: conv2d_layer3.shape:  (10, 419, 287, 32)
					-----: conv2d_layer4.shape:  (10, 53, 36, 256)
					-----: conv2d_layer5.shape:  (10, 27, 18, 1)
					-----: output.shape:  (10, 27, 18, 1)
				c. 运行内存分析
					① cpu消耗：
					② GPU消耗：
					③ 运行时间：

			Step2.2.2 生成网络					
				a. 网络深度
				b. 网络参数信息 [tf.Tensor,dtype=float32]
				[more info] Javice: GeneratorNet
					-----: input_x.shape :  (10, 837, 574, 3)
					-----: conv2d_layer1.shape:  (10, 419, 287, 32)
					-----: conv2d_layer2.shape:  (10, 210, 144, 64)
					-----: conv2d_layer3.shape:  (10, 105, 72, 128)
					-----: conv2d_layer4.shape:  (10, 53, 36, 256)
					-----: conv2d_layer5.shape:  (10, 27, 18, 512)
					-----: conv2d_layer6.shape:  (10, 14, 9, 1024)
					-----: conv2d_layer7.shape:  (10, 7, 5, 512)
					-----: conv2d_layer8.shape:  (10, 4, 3, 512)
------------------------------------------------------------------------------------
					-----: dconv2d_layer1.shape:  (10, 7, 5, 512)
					-----: dconv2d_layer2.shape:  (10, 14, 9, 1024)
					-----: dconv2d_layer3.shape:  (10, 27, 18, 512)
					-----: dconv2d_layer4.shape:  (10, 53, 36, 256) 
					-----: dconv2d_layer5.shape:  (10, 105, 72, 128)
					-----: dconv2d_layer6.shape:  (10, 210, 144, 64)
					-----: dconv2d_layer7.shape:  (10, 419, 287, 32)
					-----: dconv2d_output.shape:  (10, 837, 574, 3)
				c. 运行内存分析					
					① cpu消耗：
					② GPU消耗：
					③ 运行时间：

	Step2.2 数据准备
		
		test样本: 1024*1280*3
		train样本： 3348*4600*3
		缩放因子w_factor = 3.2695125
		缩放因子h_factor = 3.59375
		网络输入： (None, 837, 574, 3)  #None mini-batch-size 

		WTF! windows_size?? 面对不规则图片比例，该如何选择windows_size

		- 训练数据：
			- 训练集数据规模<100(小批量规模数据集的训练问题)
			- 训练样本尺寸
		- 测试数据
			- <b>测试数据运行时间分析</b>
			- 测试样本尺寸
	
	Step2.3 训练细节

		样本MSE误差计算：
			

	Step2.4 优化调参
	
	Step2.5 测试运行
	
	Step2.6 任务效果评估

"遇到的麻烦!":
	1.残差网络如何用在生成网络中~~~
	2.	# 是否要求最后需要一层全连接网络
		# output_shape = net_17.get_shape().as_list()
		# nodes = output_shape[1]*output_shape[2]*output_shape[3]
		# output = tf.reshape(output,[-1,nodes])
		# print("		-----: output.shape: ",output.shape)

Reference:
    ![SR tensorflow code](https://github.com/tegg89/SRCNN-Tensorflow)
    ![carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
    !Image Super resolutionImage Super-Resolution by Neural Texture Transfer](https://arxiv.org/abs/1903.00834v1)
    ![Progressive Generative Adversarial Networks for
Medical Image Super resolution](https://arxiv.org/)

	
	ps: 需要内存特别有底气！

`Step3`: 图像局部细节突出：

	描述：局部图像细节归属于目标检测，对待突出区域进行提取，通过回归融合边界

	demo model: 预训练好的Vgg.model
	train Sample:
	image:(m,n,3) ; Feature region: boundbox{(x1,y2),(x2,y2),...,(xn,yn)}
	test Sample:
	image:(m,n,3) ; Feature region: predict_boundbox{(predict_x1,predict_y1),...,(predict_xn,predict_yn)}

Reference:
	![Progressive Generative Adversarial Networks for
	Medical Image Super resolution](https://arxiv.org/)


开发过程中遇到的坑：

### catvsdog在搭建中遇到的问题小结

	a. 解决tensorflow报错 Attempting to use uninitialized value 
	   在代码执行之前添加： tf.reset_default_graph()
	b.: Tensor("layer_1_conv2d_weight:0", shape=(3, 3, 3, 16), dtype=float32_ref) must be from the same graph as Tensor("image_data_input:0", shape=(10, 64, 64, 3), dtype=float32).
	c. TypeError: Fetch argument 12.47665 has invalid type <class 'numpy.float32'>, must be a string or Tensor. (Can not convert a float32 into a Tensor or Operation.)这个报错不是数据feed的问题，而是等号左边的g_loss经过一次运算后得到了数值结果，覆盖了原来的g_loss操作，tf处理不了这种命名冲突，所以给变量用名字要注意啊
	d. cross_entropy = tf.nn.softmax(y)
	   print(y_.shape,cross_entropy.shape)
	   loss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(cross_entropy,1e-10,1.0)))
	 得到结果：
	 	(10, 1) (10, 2)
		Tensor("loss_function/Neg:0", shape=(), dtype=float32)
	e. 在展示tensorboard中，若无数据，则切换到相应目录下，执行：
		tensorboard --logdir=logs


卷积核shape:
	(10, 64, 64, 16)
	(10, 32, 32, 16)
	(10, 32, 32, 64)
	(10, 16, 16, 64)
	(10, 16, 16, 128)
	(10, 16, 16, 256)
	(10, 16, 16, 64)
	(10, 8, 8, 64)
	4096