#SAN16602715 Jacqueline Sangster
#Imports
from keras.datasets import fashion_mnist
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans
###
##This code is a mess, put it into functions so it can actually be fucking used.
###

#Load the training stuff. How'd you do that on keras?
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)
#If I remember rightly, the x is the image, the y is the label
#select one just to see whats going on. Expand to do all the train data if I'm understanding rightly.


sift = cv2.xfeatures2d.SIFT_create()

#detect descriptors and keypoints for each image
descList = []
#for each in x_train:
for i in range(100):
    point, descriptor = sift.detectAndCompute(x_train[0],None)
    descList.append((descriptor))

descriptors = descList[0][1]
for descriptor in descList[1:]:
    descriptors = np.vstack((descriptors, descriptor))

codebook, trash = kmeans(descriptors, 100)
print('codebook done')

#again for each image, get the histogram against the codebook?
for each in x_train:
    point, descriptor = sift.detectAndCompute(each, None)
    hist = np.histogram(descriptor, codebook)


