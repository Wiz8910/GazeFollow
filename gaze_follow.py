
import os
from collections import namedtuple

import tensorflow as tf

import scipy.io
import numpy as np
from PIL import Image

import display

# Silence Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

""" 
README_dataset.txt is incorrect
bounding_box: [x_min, y_min, x_max, y_max],
gaze: [x, y]
eye_center: [x, y]
label: int from range [1 - 25] representing grid cell id
"""
ImageAnnotation = namedtuple('ImageAnnotation',
                             ['id', 'file_path', 'bounding_box', 'gaze', 'eye_center', 'label'])

GRID_SIZE = 5 # 5x5 grid

def grid_label(file_path, xy_coordinates):
    """
    Return the classification label for the grid cell containing the coordinates.

    Grid Cell Labels
    [[1, ... 21],
      ...
     [5, ... 25]]
    """
    image = np.array(Image.open(file_path), dtype=np.uint8)
    print(image.shape)
    image_height, image_width, _ = image.shape

    x, y = xy_coordinates
    y *= image_height
    x *= image_width

    cell_width = image_width / GRID_SIZE
    cell_height = image_height / GRID_SIZE

    label_col = None
    label_row = None

    for j in range(1, GRID_SIZE+1):
        col = j * cell_width
        if x <= col:
            label_col = j
            break

    for j in range(1, GRID_SIZE+1):
        row = j * cell_height
        if y <= row:
            label_row = j
            break

    return GRID_SIZE * (label_row - 1) + label_col

def image_annotations(annotations_file_path, data_file_path):
    with open(annotations_file_path, 'r') as f:
        i = 0
        for line in f:
            line = line.split(",")

            if len(line) != 12:
                raise Exception('Annotation error')

            floats = [float(x) for x in line[1:10]]

            file_path = os.path.join(data_file_path, line[0])
            gaze = floats[7:]
            label = grid_label(file_path, gaze)

            yield ImageAnnotation(id=floats[0],
                                  file_path=file_path,
                                  bounding_box=floats[1:5],
                                  gaze=gaze,
                                  eye_center=floats[5:7],
                                  label=label)
            i += 1
            if i == 1:
                break # remove this

def image_data(annotations_file_path, data_file_path):

    annotations = [a for a in image_annotations(annotations_file_path, data_file_path)]

    # display images
    # for annotation in annotations:
    #     display.image_with_annotation(annotation, GRID_SIZE)

    file_paths = [a.file_path for a in annotations]
    filename_queue = tf.train.string_input_producer(file_paths)

    reader = tf.WholeFileReader()
    key, images = reader.read(filename_queue)

    decoded_images = tf.image.decode_jpeg(images)

    return key, decoded_images, annotations

if __name__ == '__main__':
    image_data(TRAINING_FILE_PATH, DATA_FILE_PATH)