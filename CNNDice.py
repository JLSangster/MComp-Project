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

epochs = 5
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


for i in range(1,6):
    for j in range(1,6):
        if (j == 1 & j != i):
            #start timer
            startTime = time.time()
            cnn.fit_generator(fold1, steps_per_epoch=steps, epochs=epochs, verbose = 1)
            #end timer
            end_time = time.time()
            #add the time to the cnnTrainTimes
            cnnTrainTimes[i] += (endTime - startTime)
            #start timeer
            startTime = time.time()
            transferNet.fit_generator(fold1, steps_per_epoch=steps, epochs=epochs, verbose = 1)
            #end timer
            endTime = time.time()
            #add the time to the transTrainTimes
            transTrainTimes[i] +=(endTime - startTime)
        elif (j == 2 & j != i):
            #start timer
             startTime = time.time()
             cnn.fit_generator(fold2, steps_per_epoch=steps, epochs=epochs, verbose = 1)
             #end timer
             end_time = time.time()
             #add the time to the cnnTrainTimes
             cnnTrainTimes[i] += (endTime - startTime)
             #start timeer
             startTime = time.time()
             transferNet.fit_generator(fold2, steps_per_epoch=steps, epochs=epochs, verbose = 1)
             #end timer
             endTime = time.time()
             #add the time to the transTrainTimes
             transTrainTimes[i] +=(endTime - startTime)
        elif (j == 3 & j != i):
             #start timer
             startTime = time.time()
             cnn.fit_generator(fold3, steps_per_epoch=steps, epochs=epochs, verbose = 1)
             #end timer
             end_time = time.time()
             #add the time to the cnnTrainTimes
             cnnTrainTimes[i] += (endTime - startTime) #start timeer
             startTime = time.time()
             transferNet.fit_generator(fold3, steps_per_epoch=steps, epochs=epochs, verbose = 1)
             #end timer
             endTime = time.time()
             #add the time to the transTrainTimes
             transTrainTimes[i] +=(endTime - startTime)
        elif (j == 4 & j != i):
             #start timer
             startTime = time.time()
             cnn.fit_generator(fold4, steps_per_epoch=steps, epochs=epochs, verbose = 1)
             #end timer
             end_time = time.time()
             #add the time to the cnnTrainTimes
             cnnTrainTimes[i] += (endTime - startTime)
             #start timeer
             startTime = time.time()
             transferNet.fit_generator(fold4, steps_per_epoch=steps, epochs=epochs, verbose = 1)
             #end timer
             endTime = time.time()
             #add the time to the transTrainTimes
             transTrainTimes[i] +=(endTime - startTime)
        elif (j == 5 & j != i):
             #start timer
             startTime = time.time()
             cnn.fit_generator(fold5, steps_per_epoch=steps, epochs=epochs, verbose = 1)
             #end timer
             end_time = time.time()
             #add the time to the cnnTrainTimes
             cnnTrainTimes[i] += (endTime - startTime) #start timeer
             startTime = time.time()
             transferNet.fit_generator(fold5, steps_per_epoch=steps, epochs=epochs, verbose = 1)
             #end timer
             endTime = time.time()
             #add the time to the transTrainTimes
             transTrainTimes[i] +=(endTime - startTime)
    if (i == 1):
        #start timer
        startTime = time.time()
        cnn.evaluate_generator(fold1, verbose = 1)
        #end timer
        endTime = time.time()
        #append the time to cnnTestTime
        cnnTestTimes[i] += (endTime - startTime)
        #add to the accuracies
        #start timer
        startTime = time.time()
        transferNet.evaluate_generator(fold1, verbose = 1)
        #end timer
        endTime = time.time()
        #append the time to transTestTime
        transTestTime += (endTime-startTime)
        #add to the accuracies
    elif (i == 2):
        #start timer
        startTime = time.time()
        cnn.evaluate_generator(fold2, verbose = 1)
        #end timer
        endTime = time.time()
        #append the time to cnnTestTime
        cnnTestTimes[i] += (endTime - startTime)
        #add to the accuracies
        #start timer
        startTime = time.time()
        transferNet.evaluate_generator(fold2, verbose = 1)
        #end timer
        endTime = time.time()
        #append the time to transTestTime
        transTestTime += (endTime-startTime)
        #add to the accuracies
    elif (i == 3):
        #start timer
        startTime = time.time()
        cnn.evaluate_generator(fold3, verbose = 1)
        #end timer
        endTime = time.time()
        #append the time to cnnTestTime
        cnnTestTimes[i] += (endTime - startTime)
        #add to the accuracies
        #start timer
        startTime = time.time()
        transferNet.evaluate_generator(fold3, verbose = 1)
        #end timer
        endTime = time.time()
        #append the time to transTestTime
        transTestTime += (endTime-startTime)
        #add to the accuracies
    elif (i == 4):
        #start timer
        startTime = time.time()
        cnn.evaluate_generator(fold4, verbose = 1)
        #end timer
        endTime = time.time()
        #append the time to cnnTestTime
        cnnTestTimes[i] += (endTime - startTime)
        #add to the accuracies
        #start timer
        startTime = time.time()
        transferNet.evaluate_generator(fold4, verbose = 1)
        #end timer
        endTime = time.time()
        #append the time to transTestTime
        transTestTime += (endTime-startTime)
        #add to the accuracies
    elif (i == 5):
        #start timer
        startTime = time.time()
        cnn.evaluate_generator(fold5, verbose = 1)
        #end timer
        endTime = time.time()
        #append the time to cnnTestTime
        cnnTestTimes[i] += (endTime - startTime)
        #add to the accuracies
        #start timer
        startTime = time.time()
        transferNet.evaluate_generator(fold5, verbose = 1)
        #end timer
        endTime = time.time()
        #append the time to transTestTime
        transTestTime += (endTime-startTime)
        #add to the accuracies
