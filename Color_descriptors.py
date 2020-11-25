from sys import platform as sys_pf # Fix
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import matplotlib.image as mpimg
import pywt
from skimage import *
import mahotas as mt
import glob
import csv
from skimage.color import rgb2hsv

import cv2 as cv

from skimage.feature import hog
from skimage import data, exposure

Images = sorted(glob.glob("/Users/Sebastien/Documents/3A/Optimisation/img/*.jpg"))

data = np.load('Patchs.npy')

def color_stats(image): # R_mean, G_mean, B_mean,R_std, G_std,B_std
	(means, stds) = cv.meanStdDev(image)

	stats = np.concatenate([means, stds]).flatten()
	return stats

def color_stats_HSV(image): # R_mean, G_mean, B_mean,R_std, G_std,B_std
	hsv_img = rgb2hsv(image)
	(means, stds) = cv.meanStdDev(image)

	stats_HSV = np.concatenate([means, stds]).flatten()
	return stats_HSV

def stats_HSV(image):

	imgHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)

	# H = list()
	# S = list()
	# V = list()
	H_patch = imgHSV[:,:,0] #couleur
	S_patch = imgHSV[:,:,1] #saturation
	V_patch = imgHSV[:,:,2] #brillance
	descriptors = list()

	H_mean = np.mean(H_patch)
	S_mean = np.mean(S_patch)
	V_mean = np.mean(V_patch)
	H_std = np.std(H_patch)
	S_std = np.std(S_patch)
	V_std = np.std(V_patch)

	descriptors.append([H_mean,S_mean,V_mean,H_std,S_std,V_std])



	# for i in range(21) : 
	# 	for j in range(21): 

	# 		H_patch = imgHSV[:,:,0] #couleur
	# 		S_patch = imgHSV[:,:,1] #saturation
	# 		V_patch = imgHSV[:,:,2] #brillance
	# 		H.append(H_patch)
	# 		S.append(S_patch)
	# 		V.append(V_patch)


	# moy_H = sum(H)/len(H)
	# moy_S = sum(S)/len(S)
	# moy_V = sum(V)/len(S)

	#ecart type
	# (means, stds) = cv.meanStdDev(image)
	# stats_HSV = np.concatenate([means, stds]).flatten()
	# return stats_HSV


	return descriptors

RGB_means = list()
RGB_sd = list()




img=mpimg.imread('ISIC_0000016.jpg')
b = img[data[0][2][0][0]:data[0][2][1][0],data[0][2][0][1]:data[0][2][1][1]]

print(color_stats(b))
print(color_stats_HSV(b))
print(stats_HSV(b))

#plt.imshow(rgb2hsv(b))
#plt.show()


fd, hog_image = hog(b, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, multichannel=True)

print(fd)




