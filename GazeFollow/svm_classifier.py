
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import sklearn
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib

from GazeFollow.image_dataset import Dataset
import GazeFollow.annotation as annotation
import GazeFollow.evaluation as evaluation

def svm_classifier(image_file_path, data_file_path, inception_model_path):

    data_func = annotation.image_annotations(image_file_path, data_file_path,
                                             dataset_size=1000, equal_class_proportions=True)
    data = [(a.file_path, a.gaze_label) for a in data_func]
    file_paths, gaze_labels = zip(*data)

    features = image_features(inception_model_path, file_paths)
    print("Finished extracting image features.")
    train_svm_classifier(features, gaze_labels)

def image_features(model_path, image_paths):
    """
    Finds image features using model at model_path and images at image_paths.
    Args:
        model_path: file path to inception model
        image_paths: list of file paths to jpeg images
    Return: 2-d numpy array with shape (len(image_paths), 2048)
    """

    # Load the inception model
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    print("Finished loading model.")

    features = np.empty((len(image_paths), 2048))

    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for i, image_path in enumerate(image_paths):
            image_data = gfile.FastGFile(image_path, 'rb').read()
            feature = sess.run(flattened_tensor, {'DecodeJpeg/contents:0': image_data})
            features[i, :] = np.squeeze(feature)

    return features

def train_svm_classifier(features, labels):
    """
    Train support vector machine gaze classification model.
    Prints evaluation metrics, including euclidean distance.
    Args:
        features: numpy image data from output of image_features
        labels: list of labels associated with the input features
    """
    # save 20% of data for performance evaluation
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2)

    print("Finished reading data. Starting SVM")

    parameters = [
        {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
        {"kernel": ["rbf"], "C": [1, 10, 100, 1000], "gamma": [1e-2, 1e-3, 1e-4, 1e-5]}
    ]

    svm = SVC(probability=True)

    # 10-fold cross validation
    clf = GridSearchCV(svm, parameters, cv=10, n_jobs=4)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    print("\nEvaluation Metrics:")
    print(classification_report(y_test, y_predict))

    # Print mean euclidean distance
    dist = lambda g, p: evaluation.euclidean_dist(g, p, using_tf=False)
    mean_euclidean_distance = sum(dist(g, p) for g, p in zip(y_test, y_predict)) / len(y_predict)
    print("Euclidean Distance: {}".format(mean_euclidean_distance))
