#__author__ = ritvikareddy2
#__date__ = 2018-12-11

# import cv2
# import dlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from keras.utils import np_utils
# from skimage.feature import hog
from tensorflow.python.lib.io import file_io

BUCKET_NAME = 'gs://fer-cs6140-data'

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


def read_file_from_gcs(file_name):
    with file_io.FileIO(os.path.join(BUCKET_NAME, file_name), mode='rb') as input_f:
        with file_io.FileIO(file_name, mode='w+') as output_f:
            output_f.write(input_f.read())


# takes the 'x' output of process_data as input
# def extract_hog_features(images_array):
#     hog_features_res = []
#     for img in images_array:
#         hog_feature = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
#         hog_features_res.append(hog_feature)
#     # print(len(hog_features_res), len(hog_features_res[0]))
#     return np.asarray(hog_features_res)
#
#
# # takes the 'x' output of process_data as input
# def extract_landmark_features(images_array, flatten=False):
#     face_landmarks_res = []
#     # load detector and predictors from dlib
#
#     predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#
#     # since our images are already cropped to contain only the face, we don't need to use a detector
#     rect = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
#
#     # detect face using openCV i.e cv2
#     for img in images_array:
#         cv2.imwrite('temp.jpg', img)
#         saved_img = cv2.imread('temp.jpg', 0)
#         face_landmarks = predictor(saved_img, rect[0])
#         face_landmarks_pixels = np.array([[point.x, point.y] for point in face_landmarks.parts()])
#         # print(face_landmarks)
#         face_landmarks_res.append(face_landmarks_pixels)
#     if flatten:
#         face_landmarks_res = [res.flatten() for res in face_landmarks_res]
#     return np.asarray(face_landmarks_res)


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(
                os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def plot_losses(job_dir, model_history, file_name):
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
    fig.savefig(file_name)
    copy_file_to_gcs(job_dir, file_name)

