#SAN16602715
##CNN solution for dice object classification problem

#imports
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
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
steps = 2100/batchSize

cnnTrainTimes = [0,0,0,0,0]
cnnTestTimes = [0,0,0,0,0]
cnnAccuracies = []
transTrainTimes = [0,0,0,0,0]
transTestTimes = [0,0,0,0,0]
transAccuracies = []

fold = 1

print("Training in progress")
for i in range(0,fold):
    print(i+1)
    cnn = load_model("cnn.h5")
    for j in range(0,5):
        if j != i:
            print("Training")
            startTime=time.time()
            cnn.fit_generator(folds[j], steps_per_epoch=steps, epochs=epochs)#, verbose = 0)
            endTime=time.time()
            cnnTrainTimes[i] += (endTime - startTime)
    print("Testing")
    startTime = time.time()
    loss, acc = cnn.evaluate_generator(folds[i])#, verbose = 0)
    endTime = time.time()
    cnnTestTimes[i] += (endTime - startTime)
    cnnAccuracies.append(acc)

cnnMeanTrain = sum(cnnTrainTimes)/fold
cnnMeanTest = sum(cnnTestTimes)/fold
cnnMeanAcc = sum(cnnAccuracies)/fold

print("----CNN----")
print("Mean Accuracy: " + str(cnnMeanAcc))
print("Mean Train Time: " + str(cnnMeanTrain))
print("Mean Test Time: " + str(cnnMeanTest))
print("----Each fold----")
print("Accuracies: " + str(cnnAccuracies))
print("Training Times: " + str(cnnTrainTimes))
print("Testing Times: " + str(cnnTestTimes))
print("/n")

for i in range(0,5):
    print(i+1)
    transferNet = load_model("transferNet.h5")
    for j in range(0,5):
        if j != i:
            print("Training")
            startTime=time.time()
            transferNet.fit_generator(folds[j], steps_per_epoch=steps, epochs=epochs, verbose = 0)
            endTime=time.time()
            transTrainTimes[i] += (endTime - startTime)
    print("Testing")
    startTime = time.time()
    loss, acc = transferNet.evaluate_generator(folds[i], verbose = 0)
    endTime = time.time()
    transTestTimes[i] += (endTime - startTime)
    transAccuracies.append(acc)
    
transMeanTrain = sum(transTrainTimes)/5
transMeanTest = sum(transTestTimes)/5
transMeanAcc = sum(transAccuracies)/5

print("----Transfer Net----")
print("Mean Accuracy: " + str(transMeanAcc))
print("Mean Train Time: " + str(transMeanTrain))
print("Mean Test Time: " + str(transMeanTest))
print("----Each Fold----")
print("Accuracies: " + str(transAccuracies))
print("Training Times: " + str(transTrainTimes))
print("Testing Times: " + str(transTestTimes))
print("/n")


