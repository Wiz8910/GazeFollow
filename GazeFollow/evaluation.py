
import tensorflow as tf
import math
from GazeFollow.constants import GRID_SIZE

def euclidean_dist(ground_truth_grid_label, predicted_grid_label, using_tf=True):
    """
    Returns euclidean distance between two class labels corresponding to grid cells.
    Converts label 0-24 to an x,y coordinate of center of grid cell in range(0,1).
    """

    if using_tf:
        gt_x, gt_y = euclidean_coordinate(ground_truth_grid_label)
        p_x, p_y = euclidean_coordinate(predicted_grid_label)

        return tf.sqrt(tf.square(tf.subtract(gt_y, p_y)) + tf.square(tf.subtract(gt_x, p_x)))

    else:
        coord = lambda c: (c + 0.5) / GRID_SIZE
        xy = lambda label: (coord(label // GRID_SIZE), coord(label % GRID_SIZE))

        gt_x, gt_y = xy(ground_truth_grid_label)
        p_x, p_y = xy(predicted_grid_label)

        return math.sqrt(pow(gt_y - p_y, 2) + pow(gt_x - p_x, 2))

def euclidean_coordinate(grid_class_label):
    """ Return x and y coordinates cooresponding to grid cell classification label. """
    grid_size = tf.constant(GRID_SIZE, tf.int64)

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