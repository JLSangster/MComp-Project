#SAN16602715
#Fashion MNIST CNN
from keras.datasets import fashion_mnist
from keras import Sequential, Model, utils
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from keras.applications.inception_v3 import InceptionV3
import numpy as np
from sklearn.model_selection import StratifiedKFold
import time

(xTrain, yTrain), (xTest, yTest) = fashion_mnist.load_data()

xDat = np.concatenate((xTrain,xTest), axis=0)
print(xDat.shape)
yDat = np.concatenate((yTrain,yTest), axis=0)
print(yDat.shape)
xDat = xDat.reshape(xDat.shape[0], 28,28)
xDat = np.repeat(xDat[..., np.newaxis], 3, -1)
yDat = utils.to_categorical(yTrain, 10)
yDat = utils.to_categorical(yTest, 10)

folder = StratifiedKFold(n_splits=5, shuffle=True)

classes = 6
epochs = 3
batchSize = 64
steps = 3278/batchSize

#CNN
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size = 3, activation = 'relu', input_shape = (28,28,3)))
cnn.add(Conv2D(32, kernel_size = 5, activation = 'relu'))
cnn.add(Conv2D(32, kernel_size = 7, activation = 'relu'))
cnn.add(MaxPooling2D())
cnn.add(Conv2D(16, kernel_size = 3, activation = 'relu'))
cnn.add(Conv2D(16, kernel_size = 3, activation = 'relu'))
cnn.add(Conv2D(16, kernel_size = 3, activation = 'relu'))
cnn.add(AveragePooling2D())
print("Conv done")
cnn.add(Dense(10, activation = 'relu'))
cnn.add(Dense(10, activation = 'relu'))
cnn.add(Dense(10, activation = 'relu'))
print("Dense done")
cnn.add(Flatten())
cnn.add(Dense(classes, activation = 'softmax'))
cnn.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
print("CNN done")

inception = InceptionV3(weights ='imagenet', include_top=False, input_shape = (28, 28, 3))
tNet = inception.output
tNet = Dense(28, activation='relu')(tNet)
tNet = Flatten()(tNet)
predictions = Dense(10, activation = 'softmax')(tNet)
transferNet = Model(inputs=inception.input, outputs=predictions)
transferNet.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
transferNet.fit(xTrain, yTrain, epochs=epochs)

cnnTrainTimes = []
cnnTestTimes = []
cnnAccuracies = []
transTrainTimes = []
transTestTimes = []
transAccuracies = []

for train, test in folder.split(xDat, yDat):
    startTime = time.time()
    cnn.fit(train, epochs=epochs)
    endTime = time.time()
    cnnTrainTimes.append((endtime - startTime))
    startTime = time.time()
    transferNet.fit(train, epoch=epochs)
    endTime = time.time()
    transTrainTimes.append((endTime-startTime))
    startTime = time.time()
    loss, acc = cnn.evaluate(test)
    endTime = time.time()
    cnnTestTimes.append((endTime-startTime))
    cnnAccuracies.append(acc)
    starttime = time.time()
    loss, acc = transferNet.evaluate(test)
    endTime = time.time()
    transTesttimes.append((endTime - startTime))
    transAccuracies.append(acc)

cnnMeanTrain = sum(cnnTrainTimes)/5
transMeanTrain = sum(transTrainTimes)/5
cnnMeanTest = sum(cnnTestTimes)/5
trainsMeanTest = sum(transTestTimes)/5
cnnMeanAcc = sum(cnnAccuracies)/5
transMeanAcc = sum(transAccuracies)/5


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
print("----Transfer Net----")
print("Mean Accuracy: " + transMeanAcc)
print("Mean Train Time: " + transMeanTrain)
print("Mean Test Time: " + transMeanTest)
print("----Each Fold----")
print("Accuracies: " + transAccuracies)
print("Training Times: " + transTrainTimes)
print("Testing Times: " + transTestTimes)
print("/n")
    
