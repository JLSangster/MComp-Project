#SAN16602715 Jacqueline Sangster
#imports
import os
import random
import numpy as np
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import time

def load_images():
    print("Loading data")
    dirs = ['dice/fold1', 'dice/fold2', 'dice/fold3', 'dice/fold4', 'dice/fold5']
    imgDirs = []
    #pick 10 of each randomly from each fold.
    for each in dirs:
        for folder in os.scandir(each):
            filepath = folder
            for k in range(0,50):
                filename = random.choice(os.listdir(filepath))
                imgPath = os.path.join(filepath, filename)
                imgDirs.append([imgPath, filepath])
    np.random.shuffle(imgDirs)
    imgDirs = np.array(imgDirs)
    labels = imgDirs[:,-1]
    #need to format the labels here
    for i in range(0,len(labels)):
        labels[i] = labels[i].name
    images = []
    for each in imgDirs[:,0]:
        img = cv2.imread(each)
        images.append(cv2.imread(each))
    images=np.array((images))
    print("data Loaded")
    return images, labels

def get_features(img):
    if img.shape != (480, 480):
        img= cv2.resize(img, (480, 480))
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    corners = cv2.cornerHarris(grayimg, 5, 3, 0.01)
    edges = cv2.Canny(img,140,150)
    gKernel = cv2.getGaborKernel((21,21), 9.0, 180, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    textureSeg = cv2.filter2D(grayimg, cv2.CV_8UC3, gKernel)
    features = [thresh, corners, edges, textureSeg]
    features = np.array((features))
    
    return features

crossVal = StratifiedKFold(n_splits=5, shuffle=True)

images, labels = load_images()
tree = DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 100)

sampleNum = 3000
featureNum = 4
height = 480
width = 480

features = np.zeros((sampleNum, featureNum, height, width))
for i in range(0, sampleNum):
    feat = get_features(images[i])
    features[i,:] =(feat)
    
features = np.reshape(features, (sampleNum, (featureNum * 230400)))
print("features extracted")

trainTimes = []
testTimes = []
acc = []

i=0
for train, test in crossVal.split(features, labels):
    i += 1
    print(i)
    startTime = time.time()
    tree.fit(features[train,:], labels[train])
    endTime = time.time()
    trainTimes.append((endTime-startTime))
    startTime = time.time()
    accuracy = tree.score(features[test,:], labels[test])
    endTime = time.time()
    testTimes.append((endTime-startTime))
    acc.append(accuracy)
    
trainMeanTime = sum(trainTimes) / len(trainTimes)
testMeanTime = sum(testTimes) / len(testTimes)
meanAcc = sum(acc)/ len(acc)

print("----Computer vision Tree ---")
print("Mean Train Time: " + str(trainMeanTime))
print("Mean Test Time: " + str(testMeanTime))
print("Mean Accuracy: " + str(meanAcc))
print("----Each Fold----")
print("Train Times: " + str(trainTimes))
print("Test Times: " + str(testTimes))
print("Accuracies: " + str(acc))

