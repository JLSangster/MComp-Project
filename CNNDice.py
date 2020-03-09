#SAN16602715
##CNN solution for dice object classification problem

#imports
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.applications.inception_v3 import InceptionV3
import time

height = 128
width = 128
channels = 3
classes = 6

dataLoader = ImageDataGenerator()

fold1 = dataLoader.flow_from_directory('dice/fold1', class_mode = 'categorical', target_size = (height, width))
fold2 = dataLoader.flow_from_directory('dice/fold2', class_mode = 'categorical', target_size = (height, width))
fold3 = dataLoader.flow_from_directory('dice/fold3', class_mode = 'categorical', target_size = (height, width))
fold4 = dataLoader.flow_from_directory('dice/fold4', class_mode = 'categorical', target_size = (height, width))
fold5 = dataLoader.flow_from_directory('dice/fold5', class_mode = 'categorical', target_size = (height, width))
folds = [fold1, fold2, fold3, fold4, fold5]

epochs = 3
batchSize = 64
steps = 3278/batchSize

cnn = load_model("cnn.h5")
transferNet = load_model("transferNet.h5")

cnnTrainTimes = [0,0,0,0,0]
cnnTestTimes = [0,0,0,0,0]
cnnAccuracies = []
transTrainTimes = [0,0,0,0,0]
transTestTimes = [0,0,0,0,0]
transAccuracies = []

loss, acc = cnn.evaluate_generator(fold1)
print(acc)

for i in range(1,6):
    for j in range(1,6):
        if j != i:
            startTime=time.time()
            cnn.fit_generator(folds[j], steps_per_epoch=steps, epochs=epochs, verbose = 0)
            endTime=time.time()
            cnnTrainTimes[i] += (endTime - startTime)
            startTime = time.time()
            transferNet.fit_generator(folds[j], steps_per_epoch=steps, epochs=epochs, verbose = 0)
            endTime = time.time()
            transTrainTimes[i] += (endTime - startTime)
    startTime = time.time()
    loss, acc = cnn.evaluate_generator(folds[i], verbose = 0)
    endTime = time.time()
    cnnTestTimes[i] += (endTime - startTime)
    cnnAccuracies.append(acc)
    startTime = time.time()
    loss, acc = transferNet.evaluate_generator(folds[i], verbose = 0)
    endTime = time.time()
    transTestTimes[i] += (endTime - startTime)
    transAccuracies.append(acc)
    
cnnMeanTrain = sum(cnnTrainTimes)/5
transMeanTrain = sum(transTrainTimes)/5
cnnMeanTest = sum(cnnTestTimes)/5
trainsMeanTest = sum(transTestTimes)/5
cnnMeanAcc = sum(cnnAccuracies)/5
transMeanAcc = sum(transAccuracies)/5

print("DICE")
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
