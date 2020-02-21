#SAN16602715 Jacqueline Sangster
#imports
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans

#load a dice - need to change it to all the training dice
#filename = 'C:\\Users\\Jacqui\\Documents\\Work\\Assignments\\Fourth Year\\Project\\Code\\dice\\train\\d6\\d6_45angle_0000.jpg'
#img = cv2.imread(filename)

#aquire the datafiles?
#using just the first D6 training dir for now #this filepath will change its in the wrong folder atm
trainDir = 'dice\\train\\tempd6'
#training list
trainData = []
for i in os.listdir(trainDir):
    img = cv2.imread('dice\\train\\tempd6\\{}'.format(i))
    #img = cv2.resize(img, (480,480))
    trainData.append(img)

trainData = np.array((trainData))
print('data loaded')
#trainData = trainData.reshape(trainData.shape[0], 1, 480, 480)
descList = []
sift = cv2.xfeatures2d.SIFT_create()

maxLenX = 0
maxLenY = 0
for img in trainData:
    point, descriptor = sift.detectAndCompute(img, None)
    #ah fuck it
    if descriptor.shape[0] > maxLenX:
        maxLenX = descriptor.shape[0]
    if descriptor.shape[1] > maxLenY:
        maxLenY = descriptor.shape[1]
    descriptor = np.array((descriptor))
    descList.append('d6', descriptor)
print('extraction complete')

descList = np.array((descList))
for i in descList:
    i = np.pad(i, ((0, maxLenX - i.shape[0]), (0, maxLenY - i.shape[1])), 'constant') #is 0 ok for this?

print(descList)
##
###img = cv2.drawKeypoints(img, kp, img)
###cv2.imshow('sift dice', img)
##
###with all the descriptors
codebook = KMeans(100)
codebook.fit(descList)
##
###right. The kmeans gives us the bag of possible words.
###The histogram is used to id stuff beyond that.

