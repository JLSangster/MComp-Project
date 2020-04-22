from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam

#Model parameters
height = 128
width = 128
channels = 3
classes = 6
learningRate = 0.001
opt = Adam(learning_rate = learningRate)

#Compiling the CNN
print("building")
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size = 3, activation = 'relu', input_shape = (height, width, channels)))
cnn.add(Conv2D(32, kernel_size = 5, activation = 'relu'))
cnn.add(Conv2D(32, kernel_size = 7, activation = 'relu'))
cnn.add(MaxPooling2D())
cnn.add(Conv2D(16, kernel_size = 3, activation = 'relu'))
cnn.add(Conv2D(16, kernel_size = 3, activation = 'relu'))
cnn.add(Conv2D(16, kernel_size = 3, activation = 'relu'))
cnn.add(AveragePooling2D())
cnn.add(Dense(10, activation = 'relu'))
cnn.add(Dense(10, activation = 'relu'))
cnn.add(Dense(10, activation = 'relu'))
cnn.add(Flatten())
cnn.add(Dense(classes, activation = 'softmax'))
cnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
cnn.save("cnn.h5")

#Compiling the transfer network
inception = InceptionV3(weights ='imagenet', include_top=False, input_shape = (height,width,channels))
tNet = inception.output
tNet = Flatten()(tNet)
predictions = Dense(classes, activation = 'softmax')(tNet)
transferNet = Model(inputs=inception.input, outputs=predictions)
transferNet.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
transferNet.save("transferNet.h5")
print("All done")
