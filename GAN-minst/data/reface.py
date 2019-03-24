#!/usr/bin/env python
#-*- coding: utf-8- -*-

'''UVG下图像增强'''

from PIL import Image,ImageEnhance
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

class FaceFix(object):

	def __init__(self):
		self.uvg_root = "E:/Larisvnet/data/sample/uvg/"
		self.uvg_after_root = "E:/Larisvnet/data/sample/uvg_im/"
		self.file_path = []
		self.get_file_path(self.uvg_root)

	def get_file_path(self,uvg_root):
		file_image = os.listdir(uvg_root)
		for img in file_image:
			file_path = os.path.join(uvg_root,img)
			# print("[info] Javice: Add file ", file_path)
			self.file_path.append(file_path)

	'''a1采用图片直接增强'''
	def image_enhance(self,image,c=1.25,b=1.25,ct=1.25,s=0.15):
		'''如何根据图像的色彩度调节image.brights'''
		image = ImageEnhance.Color(image).enhance(c)
		image = ImageEnhance.Brightness(image).enhance(b)
		image = ImageEnhance.Contrast(image).enhance(ct)
		image = ImageEnhance.Sharpness(image).enhance(s)
		return image

	'''采用图像直方图均衡'''
	def image_hsv_hist(self):
		data_size = len(self.file_path)
		print("[info] Javice: All the image has {} numbers".format(data_size))
		for index in range(data_size):
			image = cv2.imread(self.file_path[index])
			image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
			H,S,V = cv2.split(image)
			V = cv2.equalizeHist(V)
			image = cv2.merge([H,S,V])
			image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
			image_save_path = os.path.join(self.uvg_after_root,"{0}.jpg".format(index))
			self.save(image,image_save_path)
			print("[info] Javice: The {0} image has alreadly save".format(index))

	def save(self,image,path):
		cv2.imwrite(path,image,[int(cv2.IMWRITE_JPEG_QUALITY),100])

	def trans_opencv_to_pyplot(self,image):
		b,g,r = cv2.split(image)
		return cv2.merge([r,g,b])
	
	def hist_rgb(self,b,g,r):
		b_hist = cv2.calcHist([b],[0],None,[265.],[0.,255.])
		plt.plot(b_hist,color='b')
		b_hist = cv2.calcHist([g],[0],None,[265.],[0.,255.])
		plt.plot(b_hist,color='g')
		b_hist = cv2.calcHist([r],[0],None,[265.],[0.,255.])
		plt.plot(b_hist,color='r')
		plt.show()

	def piefix(self,b,g,r):
		'''像素级修改'''
		bm,bn = b.shape
		gm,gn = g.shape
		rm,rn = r.shape
		all_counter = 1280*1024
		b_counter = 1
		g_counter = 1
		r_counter = 1
		for row in range(bm):
			for col in range(bn):
				if b[row][col] < 40:
					b[row][col] = 55
					b_counter += 1
		drak_prob = b_counter/all_counter
		print("一共有{}像素低于40".format(b_counter))
		print("this picture's drak num is{0:4f}".format(drak_prob))
		for row in range(gm):
			for col in range(gn):
				if g[row][col] > 80:
					g_counter += 1
		print("一共有{}像素低于40".format(g_counter))
		for row in range(rm):
			for col in range(rn):
				if r[row][col] < 40:
					r_counter += 1
		print("一共有{}像素低于40".format(r_counter))
		bf = b
		gf = g
		rf = r
		return bf,gf,rf

if __name__ == '__main__':
	ff = FaceFix()
	ff.image_hsv_hist()
	#ff.analy_image(file_path)