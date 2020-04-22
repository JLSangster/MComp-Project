#SAN16602715 Jacqueline Sangster
#Decision tree testing for dice classification
#imports
import os
import random
import numpy as np
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import time

def load_images(perClass):
    #Load perClass number of images from each class in each fold
    print("Loading data")
    dirs = ['dice/fold1', 'dice/fold2', 'dice/fold3', 'dice/fold4', 'dice/fold5']
    imgDirs = []
    #pick 10 of each randomly from each fold.
    for each in dirs:
        for folder in os.scandir(each):
            filepath = folder
            for k in range(0, perClass):
                #Get image filenames and labels
                filename = random.choice(os.listdir(filepath))
                imgPath = os.path.join(filepath, filename)
                imgDirs.append([imgPath, filepath.name])
    #Shuffle the array
    np.random.shuffle(imgDirs)
    imgDirs = np.array(imgDirs)
    #Seperate the labels from the images
    labels = imgDirs[:,-1]
    images = []                                                     
    for each in imgDirs[:,0]:
        img = cv2.imread(each)
        images.append(cv2.imread(each))
    images=np.array((images))
    print("data Loaded")
    return images, labels

def get_features(img):
    #Feature extraction
    if img.shape != (480, 480):
        img= cv2.resize(img, (480, 480))
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(grayimg, 5, 3, 0.01)
    edges = cv2.Canny(img,140,150)
    gKernel = cv2.getGaborKernel((21,21), 9.0, 180, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    textureSeg = cv2.filter2D(grayimg, cv2.CV_8UC3, gKernel)
    features = [corners, edges, textureSeg]
    features = np.array((features))
    
    return features

#Tree parameters
perClass = 200
sampleNum = (perClass * 30)
featureNum = 3
height = 480
width = 480

#Split the data
crossVal = StratifiedKFold(n_splits=5, shuffle=True)

images, labels = load_images(perClass)
#Initialise the tree
tree = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 100)

#Extract and process features
print("Extracting features")
features = np.zeros((sampleNum, featureNum, height, width), np.int8)
for i in range(0, sampleNum):
    feat = get_features(images[i])
    features[i,:] =(feat)
features = np.reshape(features, (sampleNum, (featureNum * height * width)))
print("Features extracted")

#Statistics arrays
trainTimes = []
testTimes = []
acc = []

#Train and test on the five folds
i=0
for train, test in crossVal.split(features, labels):
    i += 1
    print(i)
    print("Training")
    startTime = time.time()
    tree.fit(features[train,:], labels[train])
    endTime = time.time()
    trainTimes.append((endTime-startTime))
    print("Testing")
    startTime = time.time()
    accuracy = tree.score(features[test,:], labels[test])
    endTime = time.time()
    testTimes.append((endTime-startTime))
    acc.append(accuracy)

#Calculate the means
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

