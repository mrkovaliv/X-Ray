import numpy as np

import os

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.optimizers import SGD

import cv2

import sys


#тут роблю numpy Array X_train і роблю всі зображення розміру 406/512

pathName=""

X_train=[]
i=0    
for root, directories, filenames in os.walk('/home/volodymyr/MURA/XR_ELBOW'):
    
    for filename in filenames[:100] : 
        pathName=os.path.join(root,filename)
        
        orig_img = cv2.imread(pathName, 0)
        
        orig_img = cv2.resize(orig_img ,(128,51))
        
        X_train.append(orig_img)
        
X_train = np.array(X_train )

        
#тут роблю numpy Array Y_train
pathName=""
y_train=[]

for root, directories, filenames in os.walk('/home/volodymyr/MURA/XR_ELBOW_X'):
    
    for filename in filenames[:100]: 
        pathName=os.path.join(root,filename)
        
        file = np.loadtxt(pathName)
        
        y_train.append(file)
        
y_train = np.array(y_train )
        
np.random.seed(42)



X_train = X_train.reshape(100,128,51,1)

#print(X_train.shape)
#X_train = X_train.astype('float')
#X_train /= 255

nb_classes=2

img_rows, img_cols = 406,512
#batch_size = 32

nb_epoch = 25
Y_train = np_utils.to_categorical(y_train, nb_classes)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(128,51,1),
                 activation='relu',data_format="channels_last")) # (2)
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 activation='relu',data_format="channels_last")) # (3)
model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last')) # (4)
model.add(Dropout(0.25)) 
#model.add(Conv2D(64, (3, 3), padding='valid', activation='relu',data_format="channels_last")) # (5)
#model.add(Conv2D(64, (3, 3), activation='relu',data_format="channels_last")) # (6)
#model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last')) # (7)
#model.add(Dropout(0.25)) # Добавим слой регуляризации 
model.add(Flatten())
model.add(Dense(6528, activation='relu')) # (8)
model.add(Dropout(0.5)) # Добавим слой регуляризации 
model.add(Dense(2, activation='softmax')) # (9)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.add(Dense(50, input_dim=207872, activation="relu", kernel_initializer="normal"))
#model.add(Dense(2, activation="softmax", kernel_initializer="normal"))

#print(model.summary())

#model.compile(loss="mean_absolute_error", optimizer="SGD", metrics=["accuracy"])
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=2,epochs=5,validation_split=0.1,shuffle=True,verbose=1)

#model.fit(X_train, Y_train, batch_size=300, epochs=1, validation_split=0.1, verbose=1)
