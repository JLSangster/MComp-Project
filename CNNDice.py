#SAN16602715
##CNN solution for dice object classification problem

#imports
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.applications.inception_v3 import InceptionV3

#Using special loaders to prevent massive memory use
#use fit_generator and evaluate_generator so the model knows.
dataLoader = ImageDataGenerator()
trainSet = dataLoader.flow_from_directory('dice/train', class_mode = 'categorical')
testSet = dataLoader.flow_from_directory('dice/valid', class_mode = 'categorical')

epochs = 2
height = 256
width = 256
batchSize = 4
channels = 3
classes = 6

#Then build the pure cnn
cnn = Sequential()

#check first thing
cnn.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape = (height, width, channels)))
cnn.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
cnn.add(Conv2D(16, kernel_size = 3, activation = 'relu'))
cnn.add(MaxPooling2D())
cnn.add(Flatten())
cnn.add(Dense(classes, activation = 'softmax'))
cnn.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

inception = InceptionV3(weights ='imagenet', include_top=False, input_shape = (height,width,channels))
tNet = inception.output
tNet = Flatten()(tNet)
predictions = Dense(classes, activation = 'softmax')(tNet)
transferNet = Model(inputs=inception.input, outputs=predictions)
transferNet.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

#cross validation stuff here

#Need a timer here
#cnn.fit_generator(trainSet, steps_per_epoch=batchSize, epochs=epochs)
#Timer stop here?
#cnn.evaluate_generator()
#need timer here
transferNet.fit_generator(trainSet, steps_per_epoch=batchSize, epochs=epochs)
#timer stop here
