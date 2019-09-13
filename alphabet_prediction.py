!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
import logging
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
downloaded = drive.CreateFile({'id':'1Ks5brHA6QGAP1d_LEVx_bo8Vj7ha20AT'})
downloaded.GetContentFile('alphabet_data.csv') 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("/home/nikhil/kaggle_dataset/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv")

#=============================Train A Multilayer Perceptron Model================================
train_data = dataset.drop("0",axis=1)
target = dataset["0"]
train_data = train_data.values
target = target.values
X_data = train_data/255.0
from keras.utils import np_utils
Y_data = np_utils.to_categorical(target,26)
X = X_data
Y = Y_data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,stratify=Y)
print X_test.shape
print y_test.shape

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
model = Sequential()
model.add(Dense(512,kernel_initializer = "uniform", activation = "relu",input_dim =784)))
model.add(Dropout(0.2))
model.add(Dense(512,kernel_initializer = "uniform", activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(26,kernel_initializer = "uniform", activation = "softmax"))

from keras.callbacks import ModelCheckpoint
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='emnist.model.best.hdf5',verbose=1, save_best_only=True)
hist = model.fit(X_train, y_train, batch_size=128, epochs=10,validation_split=0.2, callbacks=[checkpointer],verbose=1, shuffle=True)

score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]
#Load the Model with the Best Classification Accuracy on the Validation Set
model.load_weights('emnist.model.best.hdf5')
# Save the best model
model.save('eminst_alph_model.h5')

#=================== Train A Convolutional Neural Network Model===============================#
train_data = dataset.drop("0",axis=1)
target = dataset["0"]
train_data = train_data.values
target = target.values
X_data = train_data/255.0
from keras.utils import np_utils
X_data = X_data.reshape(372450,28,28)
target = target.reshape(372450,1)
Y_data = np_utils.to_categorical(target,26)
X = X_data
Y = Y_data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,stratify=Y)
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)


from keras.layers import Convolution2D,MaxPooling2D,Dropout,Flatten,Dense,BatchNormalization
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
model = Sequential()
model.add(Convolution2D(32,kernel_size=(3,3),activation="relu",input_shape=(28,28,1)))
model.add(Convolution2D(364,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))

model.compile(loss="categorical_crossentropy",optimizer="Adadelta", metrics=['accuracy'])
model.fit(X_train, y_train,batch_size=128,epochs=10,verbose=1,validation_data=(X_test, y_test))

model.save('emnist_cnnalphabet_model.h5')

score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]

#ititializing stuff
from keras.models import load_model
from collection import deque
import cv2

mlp_model = load_model("/home/nikhil/Downloads/emnist_cnnalphabet_model.h5")

letters ={ 0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: '-'}

import cv2
img = cv2.imread('/home/nikhil/Desktop/rTjGR4xgc.jpg',0)
image = cv2.resize(img,(28,28))
img = np.reshape(image,[1,28,28,1])
y_pred = mlp_model.predict_classes(img,batch_size=1)
print y_pred


