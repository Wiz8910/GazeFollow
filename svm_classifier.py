
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
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib

import gaze_follow
from gaze_follow import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH
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

def train_svm_classifier(features, labels, model_output_path):
    """
    train_svm_classifier trains an SVM, 
    savse the trained SVM model and
    reports the classification performance

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
    clf = GridSearchCV(svm, param, cv=10, n_jobs=4, verbose=3)

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
    print("Labels: {0}\n".format(",".join(str(l) for l in labels)))
    print(confusion_matrix(y_test, y_predict, labels=labels))

    print("\nClassification report:")
    print(classification_report(y_test, y_predict))

def sklearn_svm_classifier():
    data_gen = gaze_follow.image_annotations(TRAINING_FILE_PATH, DATA_FILE_PATH, dataset_size=1000)
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
    train_svm_classifier(features, gaze_labels, model_output_path)

def train_svm_classifer_with_tf(training_dataset, testing_dataset):

    image_size = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH
    x = tf.placeholder(tf.float32, [None, image_size])
    W = tf.Variable(tf.zeros([image_size, 25]))
    b = tf.Variable(tf.zeros([25]))
    y = tf.matmul(x, W) + b

    gaze_y_ = tf.placeholder(tf.float32, [None, 25])

    # real_feature_column = real_valued_column(...)
    # sparse_feature_column = sparse_column_with_hash_bucket(...)

    # estimator = SVM(example_id_column='example_id',
    #                 feature_columns=[real_feature_column, sparse_feature_column],
    #                 l2_regularization=10.0
    #                 model_dir='svm_saved_models')

    # Input builders
    # def input_fn_train: # returns x, y
    #     svmC = 1
    #     regularization_loss = 0.5*tf.reduce_sum(tf.square(W)) 
    #     hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE,1]), 1 - y*y_raw))
    #     svm_loss = regularization_loss + svmC*hinge_loss
    #     train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)
    
    # def input_fn_eval: # returns x, y
    #     predicted_class = tf.sign(y_raw)
    #     correct_prediction = tf.equal(y,predicted_class)
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # estimator.fit(input_fn=input_fn_train)
    # estimator.evaluate(input_fn=input_fn_eval)
    # estimator.predict(x=x)

    svmC = 1
    regularization_loss = 0.5*tf.reduce_sum(tf.square(W)) 
    hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE,1]), 1 - y*y_raw))
    svm_loss = regularization_loss + svmC*hinge_loss
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)

    predicted_class = tf.sign(y_raw)
    correct_prediction = tf.equal(y,predicted_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:

        # initialize the variables
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(training_dataset.batch_count):
            train_images, train_gaze_labels, _ = training_dataset.next_batch()

            # train_accuracy = accuracy.eval(feed_dict={x: train_images, gaze_y_: train_gaze_labels, keep_prob: 1.0})
            # print('training batch: {}, accuracy: {}'.format(i+1, train_accuracy))

            print('training batch: {}'.format(i+1))
            train_step.run(feed_dict={x: train_images, gaze_y_: train_gaze_labels, keep_prob: 0.5})

        print("training finished")

        # Test trained model
        test_images, test_gaze_labels, _ = testing_dataset.next_batch()
        print('test accuracy %g' % accuracy.eval(feed_dict={x: test_images, gaze_y_: test_gaze_labels, keep_prob: 1.0}))


def main():
    sklearn_svm_classifier()
    # training_dataset = Dataset(TRAINING_FILE_PATH, DATA_FILE_PATH, dataset_size=1, batch_size=1)
    # testing_dataset = Dataset(TESTING_FILE_PATH, DATA_FILE_PATH, dataset_size=1, batch_size=1)
    # print("data loaded")
    # train_svm_classifer_with_tf(training_dataset, testing_dataset)

if __name__ == '__main__':
    main()