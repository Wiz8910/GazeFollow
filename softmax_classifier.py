
import argparse
import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Silence Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import gaze_follow

FLAGS = None
ANY_DIM = None

# replace with command-line args
DATA_FILE_PATH = 'data'
TRAINING_FILE_PATH = os.path.join(DATA_FILE_PATH, 'train_annotations.txt')
TESTING_FILE_PATH = os.path.join(DATA_FILE_PATH, 'test_annotations.txt')

def main(_):
    train_image_batch, train_gaze_labels, train_eye_labels = gaze_follow.image_data(TRAINING_FILE_PATH, DATA_FILE_PATH)
    test_image_batch, test_gaze_labels, test_eye_labels = gaze_follow.image_data(TESTING_FILE_PATH, DATA_FILE_PATH)

    # Create the model
    # x = tf.placeholder(tf.float32, [None, gaze_follow.IMAGE_WIDTH * gaze_follow.IMAGE_HEIGHT * gaze_follow.IMAGE_DEPTH])
    W = tf.Variable(tf.zeros([gaze_follow.IMAGE_WIDTH * gaze_follow.IMAGE_HEIGHT * gaze_follow.IMAGE_DEPTH, 25]))
    b = tf.Variable(tf.zeros([25]))
    y = tf.matmul(train_image_batch, W) + b

    # # Define loss and optimizer
    gaze_y_ = tf.placeholder(tf.float32, [None, 25])
    # # eye_y_ = tf.placeholder(tf.float32, [None, 25])

    cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gaze_y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy1)

    with tf.Session() as sess:

        # initialize the variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # image = sess.run(train_image_batch)
        # print(image)

        # print("from the train set:")
        # for i in range(20):
        #     print(sess.run(train_gaze_label_batch))

        # print("from the test set:")
        # for i in range(10):
        #     print(sess.run(test_gaze_label_batch))

        sess.run(train_step, feed_dict={gaze_y_:train_gaze_labels})

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(gaze_y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={gaze_y_: test_gaze_labels}))

        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
        sess.close()

    # sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()

    # sess.run(train_step, feed_dict={x:train_images, gaze_y_:train_gaze_labels, eye_y_:train_eye_labels})

    # Test trained model
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(gaze_y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: test_images, gaze_y_: test_gaze_labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
    help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)