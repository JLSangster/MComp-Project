#SAN16602715
#Fashion MNIST CNN
from keras.datasets import fashion_mnist
from keras import Sequential, Model, utils
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.applications.inception_v3 import InceptionV3
import numpy as np

#Load data
(xTrain, yTrain), (xTest, yTest) = fashion_mnist.load_data()

xTrain = xTrain.reshape(xTrain.shape[0], 28,28,1)
xTest =  xTest.reshape(xTest.shape[0], 28,28,1)
xTrain = np.repeat(xTrain[..., np.newaxis], 3, -1)
xTest = np.repeat(xTest[..., np.newaxis], 3, -1)
yTrain = utils.to_categorical(yTrain, 10)
yTest = utils.to_categorical(yTest, 10)

#splitting the data for cross validation
#I actually don't know how many folds but it doesn't matter
#Calc the min size of the folds - account for remaineders

#for each fold
    #randomly fill with unused data up to the minimum
#then for each of the remainders
    #randomly allocate one to each fold until all data used

#is there a function for this?

#CNN - same as from thingy - next step is to adjust for optimising
#cnn = Sequential()
#cnn.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape = (28, 28, 3)))
#cnn.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
#cnn.add(Conv2D(16, kernel_size = 3, activation = 'relu'))
#cnn.add(MaxPooling2D())
#cnn.add(Flatten())
#cnn.add(Dense(10, activation = 'softmax'))
#cnn.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
#cnn.fit(xTrain, yTrain, epochs=5)

inception = InceptionV3(weights ='imagenet', include_top=False, input_shape = (28, 28, 3))
tNet = inception.output
tNet = Dense(28, activation='relu')(tNet)
tNet = Flatten()(tNet)
predictions = Dense(10, activation = 'softmax')(tNet)
transferNet = Model(inputs=inception.input, outputs=predictions)
transferNet.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
transferNet.fit(xTrain, yTrain, epochs=5)
