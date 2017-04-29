
import os
from collections import namedtuple

import tensorflow as tf

#import scipy.io
import numpy as np
from PIL import Image
import PIL
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

def grid_label(image, xy_coordinates):
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

def image_annotations(annotations_file_path, data_file_path):

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
            if len(image.shape) !=3:
                continue

            gaze = floats[7:]
            gaze_label = grid_label(image, gaze)

            eye_center = floats[5:7]
            eye_label = grid_label(image, eye_center)

            yield ImageAnnotation(id=floats[0],
                                  file_path=file_path,
                                  bounding_box=floats[1:5],
                                  gaze=gaze,
                                  eye_center=eye_center,
                                  gaze_label=gaze_label,
                                  eye_label=eye_label)
            i += 1
            if i == 100:
               break # remove this

def image_data(annotations_file_path, data_file_path):

    annotations = [a for a in image_annotations(annotations_file_path, data_file_path)]

    # display images
    #for annotation in annotations:
     #   display.image_with_annotation(annotation, GRID_SIZE)

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
    resized_image = tf.image.resize_images(decoded_image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    image = tf.reshape(resized_image, [IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH])

    min_after_dequeue = 10000
    batch_size = 100
    capacity = min_after_dequeue + 3 * batch_size
    image_batch = tf.train.batch([image], batch_size=batch_size, capacity=capacity)

    return image_batch, gaze_labels, eye_labels

