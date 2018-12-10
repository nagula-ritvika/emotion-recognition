#__author__ = ritvikareddy2
#__date__ = 2018-12-09

import cv2
import dlib
import itertools
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils
from skimage.feature import hog


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


# takes the 'x' output of process_data as input
def extract_hog_features(images_array):
    hog_features_res = []
    for img in images_array:
        hog_feature = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        hog_features_res.append(hog_feature)
    # print(len(hog_features_res), len(hog_features_res[0]))
    return np.asarray(hog_features_res)


# takes the 'x' output of process_data as input
def extract_landmark_features(images_array, flatten=False):
    face_landmarks_res = []
    # load detector and predictors from dlib
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # since our images are already cropped to contain only the face, we don't need to use a detector
    rect = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]

    # detect face using openCV i.e cv2
    for img in images_array:
        cv2.imwrite('temp.jpg', img)
        saved_img = cv2.imread('temp.jpg', 0)
        face_landmarks = predictor(saved_img, rect[0])
        face_landmarks_pixels = np.array([[point.x, point.y] for point in face_landmarks.parts()])
        # print(face_landmarks)
        face_landmarks_res.append(face_landmarks_pixels)
    if flatten:
        face_landmarks_res = [res.flatten() for res in face_landmarks_res]
    return np.asarray(face_landmarks_res)


def convert_probs_to_labels(y_probs, labels):
    predicted_label = lambda x: labels[x.argmax()]
    return [predicted_label(each) for each in y_probs]


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return fig


def test_for_single_image(file_name, model, labels):
    image_pixels = cv2.imread(file_name, 0)
    #     print(len(image_pixels))
    image_input = image_pixels.reshape(-1, 48, 48, 1)
    predicted_emotion = model.predict(image_input, verbose=0)
    #     print(predicted_emotion)
    print("This image is classified to be {} with {:.2f}% confidence".format(
        labels[predicted_emotion.argmax()], predicted_emotion.max() * 100))
    return image_pixels, predicted_emotion
