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

from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2grey


data = np.load('Patchs.npy')

img=mpimg.imread('ISIC_0000016.jpg')
plt.imshow(img)

Cropped_list = []

b = img[data[0][2][0][0]:data[0][2][1][0],data[0][2][0][1]:data[0][2][1][1]]

# plt.imshow(b)
# plt.show()

# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html##

def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean

features = extract_features(b)
print(features)

# 1 - Angular Second Moment:
# 2 - Contrast:
# 3 - Correlation:
# 4 - Variance:
# 5 - Inverse Difference Moment:
# 6 - Sum Average:
# 7 - Sum Variance:
# 8 - Sum Entropy:
# 9 - Entropy:
# 10 - Difference Variance:
# 11 - Difference Entropy:
# 12 - Information of Correlation:
# 13 - Information of Correlation:

# http://www.iosrjournals.org/iosr-jce/papers/Vol15-issue1/C01511217.pdf

