#SAN16602715
#Fashion MNIST CNN
from keras.datasets import fashion_mnist
from keras import utils, load_model
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
import time

(xTrain, yTrain), (xTest, yTest) = fashion_mnist.load_data()
print("loaded")

xDatP = np.concatenate((xTrain,xTest), axis=0)
xDat = np.zeros((xDatP.shape[0],75,75))
for each in range(0, xDat.shape[0]):
    image = xDatP[each,:]
    imgR = cv2.resize(image, (75,75))
    xDat[each,:] = imgR
                 
yDat = np.concatenate((yTrain,yTest), axis=0)
xDat = xDat.reshape(xDat.shape[0], 75,75)
xDat = np.repeat(xDat[..., np.newaxis], 3, -1)
yDat = utils.to_categorical(yTrain, 10)
yDat = utils.to_categorical(yTest, 10)

folder = StratifiedKFold(n_splits=5, shuffle=True)

classes = 6
epochs = 3
batchSize = 64

cnnTrainTimes = []
cnnTestTimes = []
cnnAccuracies = []
transTrainTimes = []
transTestTimes = []
transAccuracies = []

print("Training in progress")
i=0
for train, test in folder.split(xDat, yDat):
    i += 1
    print(i)
    cnn = load_model("cnnMNIST.h5")
    startTime = time.time()
    cnn.fit(train, epochs=epochs)
    endTime = time.time()
    cnnTrainTimes.append((endTime - startTime))
    startTime = time.time()
    loss, acc = cnn.evaluate(test)
    endTime = time.time()
    cnnTestTimes.append((endTime-startTime))
    cnnAccuracies.append(acc)

cnnMeanTrain = sum(cnnTrainTimes)/5
cnnMeanTest = sum(cnnTestTimes)/5
cnnMeanAcc = sum(cnnAccuracies)/5

print("Fashion MNIST")
print("----CNN----")
print("Mean Accuracy: " + cnnMeanAcc)
print("Mean Train Time: " + cnnMeanTrain)
print("Mean Test Time: " + cnnMeanTest)
print("----Each fold----")
print("Accuracies: " + cnnAccuracies)
print("Training Times: " + cnnTrainTimes)
print("Testing Times: " + cnnTestTimes)
print("/n")

i=0
for train, test in folder.split(xDat, yDat):
    i += 1
    print(i)
    transferNet = load_model("transferNetMNIST.h5")
    startTime = time.time()
    transferNet.fit(train, epoch=epochs)
    endTime = time.time()
    transTrainTimes.append((endTime-startTime))
    starttime = time.time()
    loss, acc = transferNet.evaluate(test)
    endTime = time.time()
    transTestTimes.append((endTime - startTime))
    transAccuracies.append(acc)

transMeanTrain = sum(transTrainTimes)/5
transMeanTest = sum(transTestTimes)/5
transMeanAcc = sum(transAccuracies)/5

print("----Transfer Net----")
print("Mean Accuracy: " + transMeanAcc)
print("Mean Train Time: " + transMeanTrain)
print("Mean Test Time: " + transMeanTest)
print("----Each Fold----")
print("Accuracies: " + transAccuracies)
print("Training Times: " + transTrainTimes)
print("Testing Times: " + transTestTimes)
print("/n")
    
