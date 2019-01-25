import os

import numpy as np
import cv2

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

dataset_root_folder = "/media/andrew/b84c4d95-450e-4802-b12d-b33e25343b1b/home/andrew/MURA_PROCESSED"

# img_rows, img_cols = 406, 512
img_rows, img_cols = 203, 256

nb_classes = 2
nb_epoch = 15
batch_size = 32

# Read all images and store as X_train and X_valid
X_train = []
X_valid = []

for root, directories, filenames in os.walk(os.path.join(dataset_root_folder, "train", "XR_ELBOW", "X")):
    for filename in filenames:
        pathName = os.path.join(root, filename)
        orig_img = cv2.imread(pathName, 0)
        orig_img = cv2.resize(orig_img, (img_rows, img_cols))

        X_train.append(orig_img)


for root, directories, filenames in os.walk(os.path.join(dataset_root_folder, "valid", "XR_ELBOW", "X")):
    for filename in filenames:
        pathName = os.path.join(root, filename)
        orig_img = cv2.imread(pathName, 0)
        orig_img = cv2.resize(orig_img, (img_rows, img_cols))

        X_valid.append(orig_img)

X_train = np.array(X_train)
X_valid = np.array(X_valid)

#  read labels for images
y_train = []
y_valid = []
for root, directories, filenames in os.walk(os.path.join(dataset_root_folder, "train", "XR_ELBOW", "Y")):
    for filename in filenames:
        pathName = os.path.join(root, filename)

        file = np.loadtxt(pathName)
        y_train.append(file)

for root, directories, filenames in os.walk(os.path.join(dataset_root_folder, "valid", "XR_ELBOW", "Y")):
    for filename in filenames:
        pathName = os.path.join(root, filename)

        file = np.loadtxt(pathName)
        y_valid.append(file)

y_train = np.array(y_train)
y_valid = np.array(y_valid)

print("X(test): ", X_train.shape)
print("Y(test): ", y_train.shape)
print("X(valid): ", X_valid.shape)
print("Y(valid): ", y_valid.shape)

np.random.seed(42)

X_train = X_train.reshape(-1, img_rows, img_cols, 1)
y_train = np_utils.to_categorical(y_train, nb_classes)

X_valid = X_valid.reshape(-1, img_rows, img_cols, 1)
y_valid = np_utils.to_categorical(y_valid, nb_classes)

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(64, 64), padding='valid', input_shape=(img_rows, img_cols, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(32, 32), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(16, 16), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(64, activation='relu'))  # (8)
model.add(Dropout(0.5))  # Regularization layer

model.add(Dense(16, activation='relu'))  # (8)
model.add(Dropout(0.25))  # Regularization layer

model.add(Dense(2, activation='softmax'))  # (9)
print("model.summary: ", model.summary())

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.01)
# adam = Adam(lr=0.01)

# Saving model weights after each epoch callback
filepath = "./models/simple-cnn-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Tensorboard callback
tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

callbacks_list = [checkpoint, tbCallBack]

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks_list,
          validation_data=(X_valid, y_valid), shuffle=True, verbose=1)
