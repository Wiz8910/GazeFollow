
import os
import math
import numpy as np
from PIL import Image
import tensorflow as tf

import gaze_follow
from gaze_follow import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH

class Dataset(object):
    """ Datasets for obtaining image batches and labels. """

    def __init__(self, annotations_file_path, data_file_path, dataset_size, batch_size,
                 width=IMAGE_WIDTH, height=IMAGE_HEIGHT, depth=IMAGE_DEPTH):

        self.width = width
        self.height = height
        self.depth = depth

        self.dataset_size = dataset_size
        self.batch_size = batch_size

        # Last batch may be less than batch_size
        self.batch_count = math.ceil(dataset_size / batch_size)

        # Separate (file_path, eye_label, gaze_label) tuples into lists
        data_gen = self._file_paths_and_labels(annotations_file_path, data_file_path)
        data = [list(x) for x in zip(*data_gen)]

        self._file_path_stack = data[0]
        self._eye_label_stack = data[1]
        self._gaze_label_stack = data[2]

    def next_batch(self):
        """ Returns image batch, gaze labels, and eye labels. """

        # Prevent popping from empty stacks
        self.batch_size = min(self.batch_size, len(self._file_path_stack))

        image_size = self.width * self.height * self.depth
        images = np.zeros((self.batch_size, image_size), dtype=np.uint8)

        for i in range(self.batch_size):
            file_path = self._file_path_stack.pop()
            image = Image.open(file_path)
            image = image.resize((self.width, self.height))
            image = np.array(image, dtype=np.uint8)
            images[i] = image.reshape(image_size)

        gaze_labels = np.zeros((self.batch_size, 25), dtype=np.float)
        eye_labels = np.zeros((self.batch_size, 25), dtype=np.float)

        for i in range(self.batch_size):
            gaze_label = self._gaze_label_stack.pop()
            eye_label = self._eye_label_stack.pop()

            gaze_labels[i][gaze_label] = 1
            eye_labels[i][eye_label] = 1

        return images, gaze_labels, eye_labels

    def _file_paths_and_labels(self, annotations_file_path, data_file_path):
        """
        Returns (file_path, eye_label, gaze_label) tuples until last annotation or
        specified dataset_size is met.
        """

        with open(annotations_file_path, 'r') as f:
            count = 0

            for line in f:
                line = line.split(",")

                file_path = os.path.join(data_file_path, line[0])

                # Prevent adding gray scale images to data set
                image_shape = np.array(Image.open(file_path), dtype=np.uint8).shape
                if len(image_shape) != 3:
                    continue

                eye_center = (float(line[6]), float(line[7]))
                eye_label = gaze_follow.grid_label(eye_center)
                gaze = (float(line[8]), float(line[9]))
                gaze_label = gaze_follow.grid_label(gaze)

                yield file_path, eye_label, gaze_label

                count += 1
                if count >= self.dataset_size:
                    break
