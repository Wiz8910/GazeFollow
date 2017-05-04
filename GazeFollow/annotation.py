
import os
from collections import namedtuple
import numpy as np
from PIL import Image

import constants

"""
Tuple containing image annotation data.
Note: README_dataset.txt is incorrect
    bounding_box: [x_min, y_min, x_max, y_max],
    gaze: [x, y]
    eye_center: [x, y]
    label: int from range [0-24] representing grid cell id
"""
ImageAnnotation = namedtuple('ImageAnnotation', [
    'id', 'file_path', 'bounding_box', 'gaze', 'eye_center', 'gaze_label', 'eye_label'])

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

            yield ImageAnnotation(annotation_id, file_path, bounding_box,
                                  gaze, eye_center, gaze_label, eye_label)
            i += 1
            if i == dataset_size:
               break

def grid_label(xy_coordinates):
    """
    Return the classification label for the grid cell containing the coordinates.

    Grid Cell Labels
    [[0, ... 20],
      ...
     [4, ... 24]]
    """
    x, y = xy_coordinates
    y *= constants.IMAGE_HEIGHT
    x *= constants.IMAGE_WIDTH

    cell_width = constants.IMAGE_WIDTH / constants.GRID_SIZE
    cell_height = constants.IMAGE_HEIGHT / constants.GRID_SIZE

    label_col = x // cell_width
    label_row = y // cell_height

    return int(constants.GRID_SIZE * label_row + label_col)
