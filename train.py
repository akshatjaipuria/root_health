from model.nn import network
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.regularizers import l2
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow import keras
import pandas as pd

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
#print(imgs.shape)

imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))

lab_enc = LabelEncoder()
labels = lab_enc.fit_transform(labels)

labels = np_utils.to_categorical(labels, len(lab_enc.classes_))
X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size = 0.4, stratify = labels, random_state = 0)

model = network.build(width = 64, height = 64, depth = 1, classes = len(lab_enc.classes_), reg = l2(0.0002))

opt = keras.optimizers.Adam(lr=1e-4, decay=1e-4 / 100)

model.compile(loss="binary_crossentropy", optimizer=opt ,metrics=["accuracy"])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs= 100, verbose=1)

keras.models.save_model(model = model,filepath = "trained_model.h5")

#model = keras.models.load_model("trained_model.h5")

y_pred = model.predict(X_test, batch_size=32)

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=lab_enc.classes_))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), history.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 100), history.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, 100), history.history["accuracy"], label = "train_acc")
plt.plot(np.arange(0, 100), history.history["val_accuracy"], label = "val_acc")
plt.title("Loass and Accuracy plot")
plt.xlabel("Epoches")
plt.ylabel("loss/accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

#saving the history object as csv and json
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = 'history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# save to csv: 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

random_ind = np.arange(0, y_test.shape[0])
random_ind = np.random.choice(random_ind, size = (25,), replace = False)

images = []

for ind in random_ind :
    image = np.expand_dims(X_train[ind], axis = 0)
    pred = model.predict(image)
    x = pred.argmax(axis = 1)[0]
    label = lab_enc.classes_[x]
    
    #dims of image is (1, 64, 64, 1), we use image[0] for the image array
    fin_img = (image[0] * 255).astype("uint8")
    fin_img = np.dstack([fin_img]*3)
    fin_img = cv2.resize(fin_img, (128,128))
    
    lab_clr = (0, 0, 255) if "non" in label else (0, 255, 0)
    cv2.putText(fin_img, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lab_clr, 2)
    images.append(fin_img)
    cv2.imshow("output", fin_img)
    cv2.waitKey(1000)

    
    
    
    
    