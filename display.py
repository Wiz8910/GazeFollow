
import matplotlib 
matplotlib.use('TkAgg') # macOS only
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def image_with_annotation(annotation, grid_size):

    #im = np.array(Image.open(annotation.file_path), dtype=np.uint8)
    im = annotation.image
    figure_height, figure_width, _ = im.shape

    # Create figure and axes
    _, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch for bounding box
    bb_x1, bb_y1, bb_x2, bb_y2 = annotation.bounding_box
    bb_x1 *= figure_width
    bb_x2 *= figure_width
    bb_y1 *= figure_height
    bb_y2 *= figure_height
    bb_width = bb_x2-bb_x1
    bb_height = bb_y2-bb_y1

    bounding_box = patches.Rectangle((bb_x1,bb_y1),
                                     bb_width,
                                     bb_height,
                                     linewidth=1,
                                     edgecolor='r',
                                     facecolor='none')

    # Draw grid
    plt.grid(True)

    # Create rectangle patch for label
    label = annotation.label 
    row = label // grid_size + 1
    col = label % grid_size

    label_width = figure_width / grid_size
    label_height = figure_height / grid_size

    label_x1 = label_width * (col - 1)
    label_y1 = label_height * (row - 1)

    label_box = patches.Rectangle((label_x1,label_y1),
                                  label_width,
                                  label_height,
                                  linewidth=1,
                                  edgecolor='r',
                                  facecolor='none')

    # Create a Circle patch for gaze
    gaze_x, gaze_y = annotation.gaze
    gaze_x *= figure_width
    gaze_y *= figure_height

    gaze = patches.Circle((gaze_x, gaze_y), 30, linewidth=1, edgecolor='b', facecolor='none')

    # Create a Circle patch for eye_center
    eye_x, eye_y = annotation.eye_center
    eye_x *= figure_width
    eye_y *= figure_height

    eye_center = patches.Circle((eye_x, eye_y), 30, linewidth=1, edgecolor='r', facecolor='none')

    arrow = patches.FancyArrowPatch(posA=(eye_x, eye_y), posB=(gaze_x, gaze_y))

    # Add the patches to the Axes
    ax.add_patch(bounding_box)
    ax.add_patch(label_box)
    ax.add_patch(gaze)
    ax.add_patch(eye_center)
    ax.add_patch(arrow)

    plt.show()