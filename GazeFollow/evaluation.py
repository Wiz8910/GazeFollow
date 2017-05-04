
import tensorflow as tf
import constants

def euclidean_dist(ground_truth_grid_label, predicted_grid_label):
    """convert label 0-24 to an x,y coordinate of range(0,1), take center of label."""

    gt_x, gt_y = euclidean_coordinate(ground_truth_grid_label)
    p_x, p_y = euclidean_coordinate(predicted_grid_label)

    return tf.sqrt(tf.square(tf.subtract(gt_y, p_y)) + tf.square(tf.subtract(gt_x, p_x)))

def euclidean_coordinate(grid_class_label):
    """ Return x and y coordinates cooresponding to grid cell classification label. """
    grid_size = tf.constant(constants.GRID_SIZE, tf.int64)

    x = tf.floordiv(grid_class_label, grid_size)
    x = coord_shift(x, grid_size)

    y = tf.mod(grid_class_label, grid_size)
    y = coord_shift(y, grid_size)

    return x, y

def coord_shift(coord, grid_size):
    """ Return transformed coordinate in range [0-1]. """
    onehalf = tf.constant(0.5, tf.float32)
    coord = tf.to_float(coord)
    coord = tf.add(coord, onehalf)
    coord = tf.divide(coord, tf.to_float(grid_size))
    return coord
