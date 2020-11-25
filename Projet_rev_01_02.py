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

def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean



Images = sorted(glob.glob("/Users/Sebastien/Documents/3A/Optimisation/img/*.jpg"))

data = np.load('Patchs.npy')

#Textures_header = ["Angular Second Moment","Contrast","Correlation","Variance","Inverse Difference Moment","Sum Average","Sum Variance","Sum Entropy","Difference Variance","Difference Entropy","Information_of_Correlation_1","Information_of_Correlation_2"]
Textures = list()
Classe = list()
for j in range(100):
	image = mpimg.imread(Images[j])
	for i in range(20):
		b = data[j][i]

		img_test = image[b[0][0]:b[1][0],b[0][1]:b[1][1]]
		features = extract_features(img_test)
		Textures.append(features)
		Classe.append(b[2])
		#Textures.append(b[2])
		print(j)


print(Textures[0])
print(Classe[0])

np.save("Textures",Textures) # Ne passe pas avec le header
np.save("Classe",Classe) # Ne passe pas avec le header


#np.savetxt("Textures.csv", Textures, delimiter=";", fmt='%s') # , header=header

# X_train = Textures[0:75]

# Y_train = Textures[76:99]

# X_test = Classe[0:75]

# Y_test = Classe[76:99]

# sc = StandardScaler()  
# #X_train = sc.fit_transform(X_train)  
# #X_test = sc.transform(X_test) 

# regressor = RandomForestRegressor(n_estimators=5, random_state=0)  
# regressor.fit(X_train, Y_train)  
# Y_test = regressor.predict(X_test) 

