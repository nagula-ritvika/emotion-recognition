#__author__ = ritvikareddy2
#__date__ = 2018-12-11

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from keras.utils import np_utils

from tensorflow.python.lib.io import file_io


# takes in the data frame of entire and the type of use(i.e dataset) and return processed data (x,y)
def process_data(main_df, use):
    subset_df = main_df[main_df["Usage"] == use]
    subset_df.drop(columns=['Usage'], inplace=True)
    subset_df['pixels'] = subset_df['pixels'].apply(
        lambda pixel_str: np.fromstring(pixel_str, sep=' '))
    x = np.vstack(subset_df['pixels'].values)
    x = x.reshape(-1, 48, 48, 1)
    y = subset_df['emotion'].values
    y = np_utils.to_categorical(y)

    return x, y


def read_data(gcs_path):
    print('downloading csv file from', gcs_path)
    file_stream = file_io.FileIO(gcs_path, mode='r')
    data = pd.read_csv(pd.compat.StringIO(file_stream.read()))
    print(data.head())
    return data


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(
                os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def plot_losses(job_dir, model_history):
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
    plt.show()
    fig.savefig('loss_comparison.png')
    copy_file_to_gcs(job_dir, 'loss_comparison.png')

