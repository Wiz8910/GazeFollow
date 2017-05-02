
import argparse
import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Silence Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import gaze_follow
from image_dataset import Dataset

FLAGS = None
ANY_DIM = None

# replace with command-line args
DATA_FILE_PATH = 'data'
TRAINING_FILE_PATH = os.path.join(DATA_FILE_PATH, 'train_annotations.txt')
TESTING_FILE_PATH = os.path.join(DATA_FILE_PATH, 'test_annotations.txt')

def main(_):
    # train_images, train_gaze_labels, train_eye_labels = gaze_follow.image_data(TRAINING_FILE_PATH, DATA_FILE_PATH)
    # test_images, test_gaze_labels, test_eye_labels = gaze_follow.image_data(TESTING_FILE_PATH, DATA_FILE_PATH)

    # 0.24
    # training_dataset_size = 100
    # training_batch_size = 10
    # testing_dataset_size = 50
    # testing_batch_size = 50

    # 0.05
    # training_dataset_size = 10000
    # training_batch_size = 100
    # testing_dataset_size = 2000
    # testing_batch_size = 2000

    training_dataset_size = 5000
    training_batch_size = 100
    testing_dataset_size = 500
    testing_batch_size = 500

    training_dataset = Dataset(TRAINING_FILE_PATH, DATA_FILE_PATH, training_dataset_size, training_batch_size)
    testing_dataset = Dataset(TESTING_FILE_PATH, DATA_FILE_PATH, testing_dataset_size, testing_batch_size)

    # Create the model
    image_dimension = gaze_follow.IMAGE_WIDTH * gaze_follow.IMAGE_HEIGHT * gaze_follow.IMAGE_DEPTH
    x = tf.placeholder(tf.float32, [None, image_dimension])
    W = tf.Variable(tf.zeros([image_dimension, 25]))
    b = tf.Variable(tf.zeros([25]))
    y = tf.matmul(x, W) + b

    gaze_y_ = tf.placeholder(tf.float32, [None, 25])
    eye_y_ = tf.placeholder(tf.float32, [None, 25])

    # Define loss and optimizer
    cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gaze_y_, logits=y))
    cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=eye_y_, logits=y))
    combined_entropy = cross_entropy1 + cross_entropy2
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(combined_entropy)

    with tf.Session() as sess:

        # initialize the variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # initialize the queue threads to start to shovel data
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)

        # feed_dict = {x: sess.run(train_images), gaze_y_:train_gaze_labels, eye_y_:train_eye_labels}
        # sess.run(train_step, feed_dict=feed_dict)

        for i in range(training_dataset.batch_count):
            print("training batch: {}".format(i+1))
            train_images, train_gaze_labels, train_eye_labels = training_dataset.next_batch()

            feed_dict = {x: train_images, gaze_y_:train_gaze_labels, eye_y_:train_eye_labels}
            sess.run(train_step, feed_dict=feed_dict)

        print("training completed")

        # Test trained model
        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(gaze_y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(sess.run(accuracy, feed_dict={x: sess.run(test_images), gaze_y_: test_gaze_labels}))

        test_images, test_gaze_labels, _ = testing_dataset.next_batch()

        correct_prediction = gaze_follow.euclidean_dist(tf.argmax(y, 1), tf.argmax(gaze_y_, 1))
        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(gaze_y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: test_images, gaze_y_: test_gaze_labels}))

        # stop our queue threads and properly close the session
        # coord.request_stop()
        # coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
    help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)