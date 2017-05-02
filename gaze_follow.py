
import os
from collections import namedtuple

import tensorflow as tf

#import scipy.io
import numpy as np
from PIL import Image
import PIL
import math

import display

# Silence Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

""" 
README_dataset.txt is incorrect
bounding_box: [x_min, y_min, x_max, y_max],
gaze: [x, y]
eye_center: [x, y]
label: int from range [0- 24] representing grid cell id
"""
ImageAnnotation = namedtuple('ImageAnnotation', [
    'id', 'file_path', 'bounding_box', 'gaze', 'eye_center', 'gaze_label', 'eye_label'])

GRID_SIZE = 5 # 5x5 grid
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_DEPTH = 3

def grid_label(xy_coordinates):
    """
    Return the classification label for the grid cell containing the coordinates.

    Grid Cell Labels
    [[0, ... 20],
      ...
     [4, ... 24]]
    """
    x, y = xy_coordinates
    y *= IMAGE_HEIGHT
    x *= IMAGE_WIDTH

    cell_width = IMAGE_WIDTH / GRID_SIZE
    cell_height = IMAGE_HEIGHT / GRID_SIZE

    label_col = x // cell_width
    label_row = y // cell_height

    return int(GRID_SIZE * label_row + label_col)

def euclidean_dist(ground_truth_grid_label, predicted_grid_label):
    """convert label 0-24 to an x,y coordinate of range(0,1), take center of label."""

    # gt_y = tf.to_float(tf.mod(ground_truth_grid_label, grid_size))
    # gt_y = tf.add(gt_y, onehalf)
    # gt_y = tf.divide(gt_y, grid_size)

    # p_y = tf.to_float(tf.mod(predicted_grid_label, grid_size))
    # p_y = tf.add(p_y, onehalf)
    # p_y = tf.divide(p_y, grid_size)

    # gt_x = tf.to_float(tf.floordiv(ground_truth_grid_label, grid_size))
    # gt_x = tf.add(gt_x, onehalf)
    # gt_x = tf.divide(gt_x, grid_size)

    # p_x = tf.to_float(tf.floordiv(predicted_grid_label, grid_size))
    # p_x = tf.add(p_x, onehalf)
    # p_x = tf.divide(p_x, grid_size)

    gt_x, gt_y = euclidean_coordinate(ground_truth_grid_label)
    p_x, p_y = euclidean_coordinate(predicted_grid_label)

    return tf.sqrt(tf.square(tf.subtract(gt_y, p_y) + tf.square(tf.subtract(gt_x, p_x))))

def euclidean_coordinate(grid_label):
    grid_size = tf.constant(GRID_SIZE, tf.int64)

    x = tf.floordiv(grid_label, grid_size)
    x = coord_shift(x, grid_size)

    y = tf.mod(grid_label, grid_size)
    y = coord_shift(y, grid_size)

    return x, y

def coord_shift(coord, grid_size):
    onehalf = tf.constant(0.5, tf.float32)
    coord = tf.to_float(coord)
    coord = tf.add(coord, onehalf)
    coord = tf.divide(coord, tf.to_float(grid_size))
    return coord

def image_annotations(annotations_file_path, data_file_path, dataset_size):

    with open(annotations_file_path, 'r') as f:
        i = 0
        for line in f:
            line = line.split(",")

            if len(line) != 12:
                raise Exception('Annotation error')

            floats = [float(x) for x in line[1:10]]

            file_path = os.path.join(data_file_path, line[0])
            image = np.array(Image.open(file_path), dtype=np.uint8)

            # Don't add gray scale images to data set
            if len(image.shape) != 3:
                continue

            annotation_id = floats[0]
            bounding_box = floats[1:5]
            eye_center = floats[5:7]
            eye_label = grid_label(eye_center)
            gaze = floats[7:]
            gaze_label = grid_label(gaze)

            yield ImageAnnotation(annotation_id, file_path, bounding_box, gaze, eye_center, gaze_label, eye_label)
            i += 1
            if i == dataset_size:
               break

def image_data(annotations_file_path, data_file_path):

    annotations = [a for a in image_annotations(annotations_file_path, data_file_path)]

    # display images
    # for annotation in annotations:
    #    display.image_with_annotation(annotation, GRID_SIZE)

    gaze_labels = np.zeros((len(annotations), 25), dtype=np.float)
    eye_labels = np.zeros((len(annotations), 25), dtype=np.float)

    for i, annotation in enumerate(annotations):
        gaze_labels[i][annotation.gaze_label] = 1
        eye_labels[i][annotation.eye_label] = 1

    file_paths = [a.file_path for a in annotations]
    filename_queue = tf.train.string_input_producer(file_paths)

    reader = tf.WholeFileReader()
    _, image = reader.read(filename_queue)

    decoded_image = tf.image.decode_jpeg(image)
    image = tf.image.resize_images(decoded_image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    image = tf.reshape(image, [IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH])

    min_after_dequeue = 10000
    batch_size = len(annotations)
    capacity = min_after_dequeue + 3 * batch_size
    image_batch = tf.train.batch([image], batch_size=batch_size, capacity=capacity)

    return image_batch, gaze_labels, eye_labels