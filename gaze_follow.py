
import os
from collections import namedtuple

import tensorflow as tf

# import matplotlib 
# matplotlib.use('TkAgg') # macOS only

from scipy import misc
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
"""
ImageAnnotation = namedtuple('ImageAnnotation',
                             ['id', 'file_path', 'bounding_box', 'gaze', 'eye_center', 'label'])

GRID_SIZE = 5 # Number of rows and cols

# def grid_labels(annotations):

#     for i in range(len(annotations)):
#         gaze_y, gaze_x = annotations[i].gaze
#         image = np.array(Image.open(annotations[i].file_path), dtype=np.uint8)
#         image_width, image_height = image.shape

#         cell_width = image_width / GRID_SIZE
#         cell_height = image_height / GRID_SIZE

#         label_col = None
#         label_row = None

#         for j in range(1, GRID_SIZE+1):
#             col = j * cell_width
#             row = j * cell_height
#             if gaze_x <= col:
#                 label_col = j
#             if gaze_y <= row:
#                 label_row = j
#             if gaze_x > col and gaze_y > row:
#                 break

#         label = 5 * (label_col - 1) + label_row
#         yield label

def grid_label(file_path, gaze):
    """
    [[1, 6,  11, 16],
     [2, 7,  12,],
     [3, 8,  13],
     [4, 9,  14],
     [5, 10, 15]]
    """
    image = np.array(Image.open(file_path), dtype=np.uint8)
    image_height, image_width, _ = image.shape

    gaze_x, gaze_y = gaze
    gaze_y *= image_height
    gaze_x *= image_width

    cell_width = image_width / GRID_SIZE
    cell_height = image_height / GRID_SIZE

    print(gaze_y, gaze_x, image_height, image_width, cell_width, cell_height)

    label_col = None
    label_row = None

    for j in range(1, GRID_SIZE+1):
        col = j * cell_width
        print(col)
        if gaze_x <= col:
            label_col = j
            break

    for j in range(1, GRID_SIZE+1):
        row = j * cell_height
        print(row)
        if gaze_y <= row:
            label_row = j
            break

    label = GRID_SIZE * (label_row - 1) + label_col
    print(label, label_row, label_col)
    return label
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
            if i == 2:
                break # remove this

def image_data(annotations_file_path, data_file_path):

    annotations = [a for a in image_annotations(annotations_file_path, data_file_path)]
    # labels = [l for l in grid_labels(annotations)]

    # mat = scipy.io.loadmat('data/train_annotations.mat')
    # print(mat)

    display.image_with_annotation(annotations[0], GRID_SIZE)

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