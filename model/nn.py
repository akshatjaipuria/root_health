from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K

class network:
    @staticmethod
    def build(width, height, depth, classes, reg):
        model = Sequential()
        inputShape = (height, width, depth)

        #adding first set of convolution
        model.add(Conv2D(64, (11, 11), input_shape = inputShape, padding = "same", kernal_regularizer = reg))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        #adding second set of convolution
        model.add(Conv2D(128, (5, 5), input_shape = inputShape, padding = "same", kernal_regularizer = reg))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        #adding third set of convolution
        model.add(Conv2D(256, (3, 3), input_shape = inputShape, padding = "same", kernal_regularizer = reg))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        #adding fully connnected layer
        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer = reg))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        #output layer - softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model