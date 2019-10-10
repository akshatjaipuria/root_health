from model.nn import network
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.regularizers import l2
from keras.utils import np_utils
from imutils import build_montages
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pickle
from tensorflow import keras

image_paths = list(paths.list_images("dataset"))
imgs = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64,64))

    imgs.append(img)
    labels.append(label)


imgs = np.array(imgs, dtype = "float")/255
print(imgs.shape)

imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))

lab_enc = LabelEncoder()
labels = lab_enc.fit_transform(labels)

labels = np_utils.to_categorical(labels, len(lab_enc.classes_))
X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size = 0.4, stratify = labels, random_state = 0)

model = network.build(width = 64, height = 64, depth = 1, classes = len(lab_enc.classes_), reg = l2(0.0002))

opt = keras.optimizers.Adam(lr=1e-4, decay=1e-4 / 100)

model.compile(loss="binary_crossentropy", optimizer=opt ,metrics=["accuracy"])

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs= 100, verbose=1)

keras.models.save_model(model = model,filepath = "nn_model.h5")

y_pred = model.predict(X_test, batch_size = 32)

print(classification_report(y_test, y_pred, target_names = lab_enc.classes_))
