# -*- coding: utf-8 -*-
"""Copy of Aki's Copy of Classification.ipynb

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, SeparableConv2D, Activation, BatchNormalization
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive')

# call load_data with allow_pickle implicitly set to true
data = np.load('/content/drive/My Drive/data/merge_data.npy', allow_pickle=True)
print('data loaded')

data.shape

data[0:2]

x = np.array([i[0] for i in data])
x = x.reshape(x.shape[0], 300, 640, 1)

# loaded boundaries already
# list_of_vids = []
# for i in range(len(boundaries) - 1):
#     list_of_vids.append(x[boundaries[i]] : x[boundaries[i+1]])

y = [i[1] for i in data]
y = np.array(y)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.30,
                                                    random_state=42)

#valid_x, test_x, valid_y, test_y = train_test_split(temp_x, temp_y, test_size=0.50,
                                                  #  random_state=42)

#datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,
                            # height_shift_range=0.2,
                            # shear_range=0.2, zoom_range=0.2,
                            # horizontal_flip=True, validation_split=0.1)

train_x.shape

chanDim = -1

#create model
model = Sequential()
#add model layers
model.add(SeparableConv2D(8, (7, 7), padding="same",
			input_shape=(300,640,1)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))

# CONV => RELU => POOL
model.add(SeparableConv2D(16, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))

# (CONV => RELU) * 2 => POOL
model.add(SeparableConv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))

model.add(SeparableConv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(86, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))

#add the above layers and increase features
# first set of FC => RELU layers
model.add(Flatten())
model.add(Dense(50))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# second set of FC => RELU layers
model.add(Dense(30))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# softmax classifier
model.add(Dense(2))
model.add(Activation("softmax"))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

#train the model
history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), batch_size = 200, epochs=1)

#_, acc = model.evaluate(test_x, test_y, verbose=1)
#print(acc)

#/content/drive/My Drive/data/testfinal.npy
#Load testing data
data = np.load('/content/drive/My Drive/data/testfinal_updated_3.npy', allow_pickle=True)
print('data loaded')

data.shape

x = np.array([i[0] for i in data])
x = x.reshape(x.shape[0], 300, 640, 1)

y = [i[1] for i in data]
y = np.array(y)

_, acc = model.evaluate(x, y, verbose=1)
print(acc)

#model.save('/content/drive/My Drive/data/my_model_Feb11_1.h5')
#print("model saved")

new_model = tf.keras.models.load_model('/content/drive/My Drive/data/my_model_Feb10_1.h5')

new_model.summary()
