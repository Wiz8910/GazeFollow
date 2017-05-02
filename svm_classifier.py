
import argparse
import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.platform import gfile

# Silence Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sklearn
from sklearn import model_selection, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib

import gaze_follow
from image_dataset import Dataset

# replace with command-line args
DATA_FILE_PATH = 'data'
TRAINING_FILE_PATH = os.path.join(DATA_FILE_PATH, 'train_annotations.txt')
TESTING_FILE_PATH = os.path.join(DATA_FILE_PATH, 'test_annotations.txt')

def create_graph(model_path):
    """
    create_graph loads the inception model to memory, should be called before
    calling extract_features.

    model_path: path to inception model in protobuf form.
    """
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(image_paths, verbose=False):
    """
    extract_features computed the inception bottleneck feature for a list of images

    image_paths: array of image path
    return: 2-d array in the shape of (len(image_paths), 2048)
    """
    feature_dimension = 2048
    features = np.empty((len(image_paths), feature_dimension))

    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for i, image_path in enumerate(image_paths):
            if verbose:
                print('Processing %s...' % (image_path))

            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image_path)

            image_data = gfile.FastGFile(image_path, 'rb').read()
            feature = sess.run(flattened_tensor, {
                'DecodeJpeg/contents:0': image_data
            })
            features[i, :] = np.squeeze(feature)

    return features 

def train_svm_classifer(features, labels, model_output_path):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance

    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # save 20% of data for performance evaluation
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2)

    param = [
        {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000]
        },
        {
            "kernel": ["rbf"],
            "C": [1, 10, 100, 1000],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]

    # request probability estimation
    svm = SVC(probability=True)

    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    clf = grid_search.GridSearchCV(svm, param,
            cv=10, n_jobs=4, verbose=3)

    clf.fit(X_train, y_train)

    if os.path.exists(model_output_path):
        joblib.dump(clf.best_estimator_, model_output_path)
    else:
        print("Cannot save trained svm model to {0}.".format(model_output_path))

    print("\nBest parameters set:")
    print(clf.best_params_)

    y_predict=clf.predict(X_test)

    labels=sorted(list(set(labels)))
    print("\nConfusion matrix:")
    print("Labels: {0}\n".format(",".join(labels)))
    print(confusion_matrix(y_test, y_predict, labels=labels))

    print("\nClassification report:")
    print(classification_report(y_test, y_predict))

def main():
    data_gen = gaze_follow.image_annotations(TRAINING_FILE_PATH, DATA_FILE_PATH, dataset_size=10000)
    data = [(a.file_path, a.gaze_label) for a in data_gen]
    file_paths, gaze_labels = zip(*data)

    model_path = "../inception_dec_2015/tensorflow_inception_graph.pb"
    model_output_path = "svm_model.txt"

    if not os.path.exists(model_path):
        print("Model path does not exist")
        sys.exit(0)
    if not os.path.exists(model_output_path):
        print("Model output path does not exist")
        sys.exit(0)

    create_graph(model_path)
    features = extract_features(file_paths, verbose=True)
    train_svm_classifer(features, gaze_labels, model_output_path)

if __name__ == '__main__':
    main()