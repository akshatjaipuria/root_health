
import tensorflow as tf
from tensorflow import keras
class network:
    @staticmethod
    def build(width, height, depth, classes, reg):
        model = keras.Sequential()
        inputShape = (height, width, depth)

        #adding first set of convolution
        model.add(keras.layers.Conv2D(64, (11, 11), input_shape = inputShape, padding = "same", kernel_regularizer = reg))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPool2D(pool_size = (2, 2)))
        model.add(keras.layers.Dropout(0.25))

        #adding second set of convolution
        model.add(keras.layers.Conv2D(128, (5, 5), input_shape = inputShape, padding = "same", kernel_regularizer = reg))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPool2D(pool_size = (2, 2)))
        model.add(keras.layers.Dropout(0.25))

        #adding third set of convolution
        model.add(keras.layers.Conv2D(256, (3, 3), input_shape = inputShape, padding = "same", kernel_regularizer = reg))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPool2D(pool_size = (2, 2)))
        model.add(keras.layers.Dropout(0.25))

        #adding fully connnected layer
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, kernel_regularizer = reg))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(0.5))

        #output layer - softmax classifier
        model.add(keras.layers.Dense(classes))
        model.add(keras.layers.Activation("softmax"))

        return model