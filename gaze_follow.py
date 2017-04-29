
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
ImageAnnotation = namedtuple('ImageAnnotation',
                             ['id', 'file_path', 'bounding_box', 'gaze', 'eye_center', 'label', 'image'])

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
    #image_height, image_width, _ = image.shape

    x, y = xy_coordinates
    y *= IMAGE_HEIGHT
    x *= IMAGE_WIDTH

    cell_width = IMAGE_HEIGHT / GRID_SIZE
    cell_height = IMAGE_HEIGHT / GRID_SIZE

    label_col = x//cell_width
    label_row = y//cell_height
    '''
    for j in range(1, GRID_SIZE+1):
        col = j * cell_width
        if x <= col:
            label_col = j-1
            break

    for j in range(1, GRID_SIZE+1):
        row = j * cell_height
        if y <= row:
            label_row = j-1
            break
    '''
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
            image = Image.open(file_path)
            image = image.resize((IMAGE_WIDTH,IMAGE_HEIGHT))
            image = np.array(image, dtype=np.uint8)
            #print(image.shape)
            if len(image.shape) !=3:
                continue
            gaze = floats[7:]
            label = grid_label(image, gaze)
            yield ImageAnnotation(id=floats[0],
                                  file_path=file_path,
                                  bounding_box=floats[1:5],
                                  gaze=gaze,
                                  eye_center=floats[5:7],
                                  label=label,
                                  image=image)
            i += 1
            #if i == 1000:
             #   break # remove this

def image_data(annotations_file_path, data_file_path):

    annotations = [a for a in image_annotations(annotations_file_path, data_file_path)]

    # display images
    #for annotation in annotations:
     #   display.image_with_annotation(annotation, GRID_SIZE)

    file_paths = [a.file_path for a in annotations]
    filename_queue = tf.train.string_input_producer(file_paths)

    reader = tf.WholeFileReader()
    key, images = reader.read(filename_queue)

    decoded_images = tf.image.decode_jpeg(images)

    labels = np.zeros((len(annotations), 25), dtype=np.uint8)
    for i, annotation in enumerate(annotations):
        labels[i][annotation.label] = 1

    # print(key, type(decoded_images))

    #images = [a.image for a in annotations]
    images=[]
    for a in annotations:
        image = a.image.reshape(IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH)
        images.append(image)

    return key, decoded_images, annotations,labels,images