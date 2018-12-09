#__author__ = ritvikareddy2
#__date__ = 2018-12-09

import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D, \
    Reshape
from keras.models import Sequential
from keras.utils import np_utils

from sklearn.metrics import confusion_matrix

from data_utils import process_data, convert_probs_to_labels


def load_data():
    df = pd.read_csv("fer2013.csv")
    return df





def train_and_evaluate_model():

    data = load_data()
    x_train, y_train = process_data(data, 'Training')
    x_validation, y_validation = process_data(data, 'PublicTest')

    model = Sequential()

    model.add(Conv2D(64, 3, data_format="channels_last", kernel_initializer="he_normal",
                     input_shape=(48, 48, 1), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.6))

    model.add(Conv2D(32, 3, activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, 3, activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))

    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # save best weights
    checkpoint = ModelCheckpoint(filepath='cnn_basic_2', verbose=1, save_best_only=True)

    # num epochs
    epochs = 20

    # run model
    model_history = model.fit(x_train, y_train, epochs=epochs,
                              shuffle=True,
                              batch_size=100, validation_data=(x_validation, y_validation),
                              callbacks=[checkpoint], verbose=2)

    # save model to json
    model_json = model.to_json()
    with open("cnn2.json", "w") as json_file:
        json_file.write(model_json)

    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.suptitle('CNN on raw pixel data', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(model_history.history['loss'], color='b', label='Training Loss')
    plt.plot(model_history.history['val_loss'], color='r', label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(model_history.history['acc'], color='b', label='Training Accuracy')
    plt.plot(model_history.history['val_acc'], color='r', label='Validation Accuracy')
    plt.legend(loc='lower right')
    fig.savefig("Loss_Accuracy_Comparison.png")
    plt.show()

    x_test, y_test = process_data(data, "PrivateTest")
    y_predicted_probs = model.predict(x_test, verbose=1)

    emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise",
                      6: "Neutral"}
    labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    score = model.evaluate(x_test, y_test, verbose=1)

    print("Metrics: ", model.metrics_names)
    print("Loss on test data", score[0])
    print("Test Accuracy", score[1])


    y_predicted_labels = convert_probs_to_labels(y_predicted_probs, labels)
    y_actual_labels = convert_probs_to_labels(y_test, labels)
