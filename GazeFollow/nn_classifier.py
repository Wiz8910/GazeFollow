
import argparse
import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Silence Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import evaluation

def deepnn(x):
    """
    Build the graph for a deep net for classifying labels.
    Args:
        x: an input tensor with the dimensions (N_examples, N_pixels)
    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 25), with values
        equal to the logits of classifying the label into one of 25 classes.
        keep_prob is a scalar placeholder for the probability of dropout.
    """
    # Reshape to use within a convolutional neural net.
    x_image = tf.reshape(x, [-1, 28, 28, 3])

    # First convolutional layer. Maps one RGB image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    dimension = 7 * 7 * 64

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([dimension, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, dimension])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout to prevent overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 25])
    b_fc2 = bias_variable([25])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def nn_classifier(training_dataset, testing_dataset):
    """
    Neural net classifier for gaze location in grid.
    Args:
          train_fp: Training images file path
          test_fp: Testing images file path
          data_fp: Data file path
    """
    image_size = training_dataset.width * training_dataset.height * training_dataset.depth # 2352

    print("data loaded")

    x = tf.placeholder(tf.float32, [None, image_size])

    # Define loss and optimizer
    gaze_y_ = tf.placeholder(tf.float32, [None, 25])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=gaze_y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = evaluation.euclidean_dist(tf.argmax(y_conv, 1), tf.argmax(gaze_y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:

        # initialize the variables
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(training_dataset.batch_count):
            train_images, train_gaze_labels, _ = training_dataset.next_batch()
            print('training batch: {}'.format(i+1))

            train_step.run(feed_dict={
                x: train_images,
                gaze_y_: train_gaze_labels,
                keep_prob: 0.5
            })

        print("training finished")

        # Test trained model
        test_images, test_gaze_labels, _ = testing_dataset.next_batch()
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_images,
            gaze_y_: test_gaze_labels,
            keep_prob: 1.0
        }))
