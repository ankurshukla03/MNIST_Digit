#!/usr/bin/env python
# coding: utf-8

# # CNN
# CNN model on Digit Dataset
# 
# Read the below link again 
# https://www.kaggle.com/chapagain/digit-recognizer-beginner-s-guide-mlp-cnn-keras
# https://keras.io/models/model/
# read more about fit and evaluate function
# Check the jupyter notebook for this python file in Notebook/CNN_Digit_Update.ipynb
import numpy as np
import pandas as pd

from keras.utils import np_utils
# for Convolutional Neural Network (CNN) model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D

from keras import backend as K
K.set_image_dim_ordering('th')

def baseline_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(1, 28, 28), activation='relu',data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size = 4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

        # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#Send the array value of the test image in 28,28 size
# returns the predicted digit
def prediction(newImage):
    model = baseline_model()
    model.load_weights('mnist_digit.h5')
    result = model.predict(newImage)
    result = np.argmax(result,axis=1)
    return result
