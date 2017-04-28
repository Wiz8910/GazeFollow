
import os
from collections import namedtuple

import tensorflow as tf

import scipy.io
import numpy as np
from PIL import Image

import display

# Silence Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# replace with command-line args
DATA_FILE_PATH = 'data'
TRAINING_FILE_PATH = os.path.join(DATA_FILE_PATH, 'train_annotations.txt')
TESTING_FILE_PATH = os.path.join(DATA_FILE_PATH, 'test_annotations.txt')

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
"""
SVM: 
We generate features by concatenating the quantized eye position with 
pool5 of the ImageNet-CNN [12] for both the full image and the head image. 
We train a SVM on these features to predict gaze using a similar classification 
grid setup as our model. We evaluate this approach for both, a single grid and shifted grids.

The gaze pathway 
only has access to the closeup image of the person’s head and their 
location, and produces a spatial map, G(xh, xp), of size D × D. 

The saliency pathway 
sees the full image but not the person’s location, and produces another 
spatial map, S(xi), of the same size D × D. 
We then combine the pathways with an element-wise product: 
    yˆ = F (G(xh, xp) ⊗ S(xi))
where ⊗ represents the element-wise product. 
F (·) is a fully connected layer that uses the multiplied
pathways to predict where the person is looking, yˆ.

Predict Gaze mask: 
In the gaze pathway, we use a convolutional network on the head image. We concatenate 
its output with the head position and use several fully connected layers and a final 
sigmoid to predict the D × D gaze mask.

The saliency map and gaze mask are 13 × 13 in size (i.e., D = 13), and we use 5 shifted 
grids of size 5 × 5 each (i.e., N = 5).
"""

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
            if i == 8:
                break # remove this

def image_data(annotations_file_path, data_file_path):

    annotations = [a for a in image_annotations(annotations_file_path, data_file_path)]

    for annotation in annotations:
        display.image_with_annotation(annotation, GRID_SIZE)

    # file_paths = [a.file_path for a in annotations]

    # filename_queue = tf.train.string_input_producer(file_paths)

    # reader = tf.WholeFileReader()
    # key, images = reader.read(filename_queue)

    # decoded_images = tf.image.decode_jpeg(images)

    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)

    # sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()

    # tf.image.draw_bounding_boxes(decoded_images, boxes, name=None)

    # print(my_img)

    # return Dataset()

if __name__ == '__main__':
    image_data(TRAINING_FILE_PATH, DATA_FILE_PATH)