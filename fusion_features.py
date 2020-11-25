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

from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2grey

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

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

import scipy.misc as misc
import skimage.filters
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage import measure
from skimage.filters import threshold_otsu
from scipy import ndimage
import math as m


def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean


def color_stats(image): # R_mean, G_mean, B_mean,R_std, G_std,B_std
	(means, stds) = cv.meanStdDev(image)

	stats = np.concatenate([means, stds]).flatten()
	return stats


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

	return descriptors


def HOG_descriptors(image):

	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, multichannel=True)

	return fd


###################################################################################################

# def raw_moment(data, i_order, j_order):
#     (nrows, ncols) = data.shape
#     y_indices, x_indicies = np.mgrid[:nrows, :ncols]
#     return (data * x_indicies**i_order * y_indices**j_order).sum()


# def moments_cov(data):
#     data_sum = data.sum()
#     m10 = raw_moment(data, 1, 0)
#     m01 = raw_moment(data, 0, 1)
#     x_centroid = m10 / data_sum
#     y_centroid = m01 / data_sum
#     u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
#     u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
#     u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
#     cov = np.array([[u20, u11], [u11, u02]])
#     return cov

# ## Methode 1


# def distance_centre_masse(Image,patch):
#     #img=misc.imread(Image, flatten=True)
#     centre_masse=ndimage.measurements.center_of_mass(Image)
#     centres_patch=[]
#     centre1=[]
#     centre2=[]

#     #Centre des patchs

#     for i in range (0,len(patch)-1):
#         for p in range (0,20): #20 patchs par image
#             for xy in range (0,2) : 
#                 pa = patch[i][p][0][xy] + 10
#                 if xy == 0:
#                     x = pa
#                 else :
#                     y = pa
#             centres_patch.append([x,y])
#     distance=[]
#     for i in range (len(centres_patch)):
#         distance.append(m.sqrt((centres_patch[i][0]-int(centre_masse[0]))**2+(centres_patch[i][1]-int(centre_masse[1]))**2))

#     return distance

# ## Methode 2

# def interieur_exterieur(Image,patch):
#         #Centre des patchs
#     centres_patch=[]
#     for i in range (0,len(patch)-1):
#         for p in range (0,20): #20 patchs par image
#             for xy in range (0,2) : 
#                 pa = patch[i][p][0][xy] + 10
#                 if xy == 0:
#                     x = pa
#                 else :
#                     y = pa
#             centres_patch.append([x,y])        
#     img = misc.imread(Image, flatten=True)
#     cov = moments_cov(img) #Matrice des covariances
#     evals, evecs = np.linalg.eig(cov) #Valeurs Propres, Vecteurs propres
    
#     # definir si le point est dans l'ellipse ou non 

#     ## vecteur 1
#     a1 = evecs[0][1]-evecs[0][0]
#     b1 = evecs[0][1]-a1

#     ## vecteur 2
#     a2 = evecs[1][1]-evecs[1][0]
#     b2 = evecs[1][1]-a2
    
#     # definir le centre de l'ellipse
#     mat1 = np.array([[1,-a1],[1,-a2]])
#     mat2 = np.array([b1,b2])
#     centre_ellipse = np.linalg.solve(mat1,mat2)
    
#     result=[]
#     for i in range (len(centres_patch)):
#         X= evecs[0][1]-evecs[0][0] #longeur
#         Y=evecs[1][1]-evecs[1][0] #largeur
#         val = ( (centres_patch[i][0] - centre_ellipse[0])/X )**2 + ( (centres_patch[i][1] - centre_ellipse[1])/Y )**2
#         if val < 1 :
#             result.append(1)  # le patch est dans l'ellipse
#         else :
#             result.append(0)
#     return result



    ###############################################################################






Images = sorted(glob.glob("/Users/Sebastien/Documents/3A/Optimisation/img/*.jpg"))

data = np.load('Patchs.npy')


descripteurs = list()
Classe = list()
for j in range(100):
	image = mpimg.imread(Images[j])
	for i in range(20):
		b = data[j][i]

		img_test = image[b[0][0]:b[1][0],b[0][1]:b[1][1]]
		features = extract_features(img_test)
		#descripteurs.append(features)

		color_stq = color_stats(img_test)
		#descripteurs.append(color_stq)

		HSV_stq = stats_HSV(img_test)
		#descripteurs.append(HSV_stq)

		HOG_stq = HOG_descriptors(img_test)
		#descripteurs.append([features,color_stq,HSV_stq,HOG_stq])

		#test = distance_centre_masse(b,img_test)
		#print(test)
		desc = np.concatenate((features, color_stq,HSV_stq,HOG_stq), axis=None)
		descripteurs.append(desc)
		Classe.append(b[2])
		#Textures.append(b[2])
		print(j)


print(descripteurs[0])
print(Classe[0])

np.save("Descripteurs",descripteurs) # Ne passe pas avec le header
np.save("Classe",Classe) # Ne passe pas avec le header