
import os
import math
import numpy as np
from PIL import Image
import tensorflow as tf

import GazeFollow.annotation as annotation

class Dataset(object):
    """ Datasets for obtaining image batches and labels. """

    def __init__(self, annotations_file_path, data_file_path,
                 dataset_size, batch_size,
                 width, height, depth):

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
                eye_label = annotation.grid_label(eye_center)
                gaze = (float(line[8]), float(line[9]))
                gaze_label = annotation.grid_label(gaze)

                yield file_path, eye_label, gaze_label

                count += 1
                if count >= self.dataset_size:
                    break

# def image_data(annotations_file_path, data_file_path):
#     """ Deprecated function.  Replaced with Dataset class. """

#     annotations = [a for a in annotation.image_annotations(annotations_file_path, data_file_path, dataset_size=1000)]

#     # display images
#     # for annotation in annotations:
#     #    display.image_with_annotation(annotation, GRID_SIZE)

#     gaze_labels = np.zeros((len(annotations), 25), dtype=np.float)
#     eye_labels = np.zeros((len(annotations), 25), dtype=np.float)

#     for i, annotation in enumerate(annotations):
#         gaze_labels[i][annotation.gaze_label] = 1
#         eye_labels[i][annotation.eye_label] = 1

#     file_paths = [a.file_path for a in annotations]
#     filename_queue = tf.train.string_input_producer(file_paths)

#     reader = tf.WholeFileReader()
#     _, image = reader.read(filename_queue)

#     decoded_image = tf.image.decode_jpeg(image)
#     image = tf.image.resize_images(decoded_image, [gf.IMAGE_WIDTH, gf.IMAGE_HEIGHT])
#     image = tf.reshape(image, [gf.IMAGE_WIDTH * gf.IMAGE_HEIGHT * gf.IMAGE_DEPTH])

#     min_after_dequeue = 10000
#     batch_size = len(annotations)
#     capacity = min_after_dequeue + 3 * batch_size
#     image_batch = tf.train.batch([image], batch_size=batch_size, capacity=capacity)

#     return image_batch, gaze_labels, eye_labels
