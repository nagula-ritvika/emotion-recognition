#__author__ = ritvikareddy2
#__date__ = 2018-12-11

import keras

from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D
from keras.models import Sequential, Model


def get_cnn_model_A():
    model = Sequential()
    # Layer 1
    model.add(Conv2D(64, 3, data_format="channels_last", kernel_initializer="he_normal",
                     input_shape=(48, 48, 1), activation='relu'))
    model.add(BatchNormalization())
    # Layer 2
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    # Pooling Layer
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.6))
    # Layer 3
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(BatchNormalization())
    # Layer 4
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(BatchNormalization())
    # Layer 5
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(BatchNormalization())
    # Pooling Layer
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.6))
    # Flatten
    model.add(Flatten())
    # Fully Connected Layer
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    # Output Layer
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='momentum', metrics=['accuracy'])
    model.summary()
    return model


def get_cnn_model():
    model = Model()
    # Layer 1
    model.add(Conv2D(64, 3, data_format="channels_last", kernel_initializer="he_normal",
                     input_shape=(48, 48, 1), activation='relu'))
    model.add(BatchNormalization())
    # Layer 2
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    # Pooling Layer
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.6))
    # Layer 3
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(BatchNormalization())
    # Layer 4
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(BatchNormalization())
    # Layer 5
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(BatchNormalization())
    # Pooling Layer
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.6))
    # Flatten
    model.add(Flatten())
    # Fully Connected Layer
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    # Output Layer
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def get_simple_model(model_type='A'):

    if model_type=='A':
        return get_cnn_model_A()
    elif model_type=='B':
        images_input_layer, images_last_layer = get_cnn_model_B()

    output_layer = Dense(7, activation='softmax')(images_last_layer)
    model = Model(inputs=images_input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def get_hybrid_model(model_type='A', only_landmarks=True):

    images_input_layer, images_last_layer = get_images_network(model_type)

    features_input_layer, features_last_layer = get_features_network(only_landmarks)

    # merge final outputs from both networks
    merged_net = keras.layers.merge.concatenate([images_last_layer, features_last_layer],
                                                axis=-1)
    # Output Layer
    output_layer = Dense(7, activation='softmax')(merged_net)
    model = Model(inputs=[images_input_layer, features_input_layer], outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def get_images_network(model_type='A'):

    if model_type == 'B':
        return get_cnn_model_B()
    elif model_type == 'A':
        return get_cnn_model_A()


def get_cnn_model_B():
    # cnn for just the raw pixels
    # Input layer 1
    images_nn = Input(shape=(48, 48, 1))
    # Layer 1
    images_conv1 = Conv2D(64, 3, kernel_initializer='he_normal', activation='relu')(images_nn)
    images_batchnorm1 = BatchNormalization()(images_conv1)
    images_pool1 = MaxPool2D(pool_size=(3, 3), strides=2)(images_batchnorm1)
    # Layer 2
    images_conv2 = Conv2D(128, 3, activation='relu')(images_pool1)
    images_batchnorm2 = BatchNormalization()(images_conv2)
    images_pool2 = MaxPool2D(pool_size=(3, 3), strides=2)(images_batchnorm2)
    # Layer 3
    images_conv3 = Conv2D(256, 3, activation='relu')(images_pool2)
    images_batchnorm3 = BatchNormalization()(images_conv3)
    images_pool3 = MaxPool2D(pool_size=(3, 3), strides=2)(images_batchnorm3)
    images_dropout1 = Dropout(0.6)(images_pool3)
    # Flatten
    images_flat1 = Flatten()(images_dropout1)
    # Fully Connected Layer 1
    images_dense1 = Dense(4096, activation='relu')(images_flat1)
    images_dropout2 = Dropout(0.6)(images_dense1)
    # Fully Connected Layer 2
    images_dense2 = Dense(1024, activation='relu')(images_dropout2)
    images_batchnorm4 = BatchNormalization()(images_dense2)
    # Fully Connected Layer 3
    images_dense3 = Dense(128, activation='relu')(images_batchnorm4)

    input_layer = images_nn
    last_layer = images_dense3

    return input_layer, last_layer


def get_features_network(only_landmarks=True):

    if only_landmarks:

        # input shape will be 68*2 (from landmarks)
        # Input Layer
        features_nn = Input(shape=(68, 2))

    else:
        # nn for hog + landmark features
        # input shape will be 72 (from hog) + 68*2 (from landmarks)
        features_nn = Input(shape=(208,))

    # Flatten
    features_flat1 = Flatten()(features_nn)
    # Fully Connected Layer 4
    features_dense1 = Dense(1024, activation='relu')(features_flat1)
    features_batchnorm1 = BatchNormalization()(features_dense1)
    # Fully Connected Layer 5
    features_dense2 = Dense(128, activation='relu')(features_batchnorm1)
    features_batchnorm2 = BatchNormalization()(features_dense2)

    return features_nn, features_batchnorm2
