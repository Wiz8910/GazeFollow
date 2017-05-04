
import tensorflow as tf

from image_dataset import Dataset
import evaluation
import constants

def softmax_classifier(training_dataset, testing_dataset):

    # Create the model
    image_dimension = training_dataset.width * training_dataset.height * training_dataset.depth
    x = tf.placeholder(tf.float32, [None, image_dimension])
    W = tf.Variable(tf.zeros([image_dimension, 25]))
    b = tf.Variable(tf.zeros([25]))
    y = tf.matmul(x, W) + b

    gaze_y_ = tf.placeholder(tf.float32, [None, 25])
    eye_y_ = tf.placeholder(tf.float32, [None, 25])

    # Define loss and optimizer
    cross_entropy1 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=gaze_y_, logits=y))
    cross_entropy2 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=eye_y_, logits=y))
    combined_entropy = cross_entropy1 + cross_entropy2
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(combined_entropy)

    with tf.Session() as sess:

        # initialize the variables
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(training_dataset.batch_count):
            print("training batch: {}".format(i+1))
            train_images, train_gaze_labels, train_eye_labels = training_dataset.next_batch()

            feed_dict = {x: train_images, gaze_y_:train_gaze_labels, eye_y_:train_eye_labels}
            sess.run(train_step, feed_dict=feed_dict)

        print("training completed")

        # Test trained model
        test_images, test_gaze_labels, _ = testing_dataset.next_batch()
        correct_prediction = evaluation.euclidean_dist(tf.argmax(y, 1), tf.argmax(gaze_y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: test_images, gaze_y_: test_gaze_labels}))
