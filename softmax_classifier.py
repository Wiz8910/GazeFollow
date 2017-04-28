
import argparse
import sys
import os
import numpy as np
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
    key, decoded_images, annotations = gaze_follow.image_data(TRAINING_FILE_PATH, DATA_FILE_PATH)
    labels = [annotation.label for annotation in annotations]

    print(key, type(decoded_images))

    # (512, 759, 3)

    # x = tf.placeholder(tf.float32, [None, 784])
    # W = tf.Variable(tf.zeros([784, 10]))
    # b = tf.Variable(tf.zeros([10]))

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    # W = tf.Variable([np.random.randn(), np.random.randn(), np.random.randn()])
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [ANY_DIM, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1): #length of your filename list
        image = images[i].eval()
        print(image.shape)
        Image.fromarray(np.asarray(image)).show()

    coord.request_stop()
    coord.join(threads)

    # Train
    # for _ in range(1000):
    #     batch_xs, batch_ys = mnist.train.next_batch(100)
    #     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    sess.run(train_step, feed_dict={x:decoded_images, y_:labels})

    # Test trained model
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: mnist.test.images,
    #                                     y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
    help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)