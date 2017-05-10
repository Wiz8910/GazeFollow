
import argparse
import os
import sys
import tensorflow as tf

# Silence Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import GazeFollow.constants as constants
import GazeFollow.display as display
from GazeFollow.image_dataset import Dataset
from GazeFollow.nn_classifier import nn_classifier
from GazeFollow.softmax_classifier import softmax_classifier
from GazeFollow.svm_classifier import svm_classifier

DESCRIPTION = 'Follow the gaze of a subject in an image using various computer vision methods.'

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument("data_folder_path", default='data',
                        help='Folder file path to GazeFollow training and testing image annotations.')
    parser.add_argument("-m", "--method", default='neural_net',
                        help='neural_net, softmax, or svm. Defaults to neural_net.')

    svm_help_message = 'svm_classifier uses inception model to extract features. Download it and provide file path.'
    parser.add_argument("-s", "--svm_model_file_path", default='inception_dec_2015/tensorflow_inception_graph.pb',
                        help=svm_help_message)

    args = parser.parse_args()


    # Validate input
    if not os.path.exists(args.data_folder_path):
        print("Invalid data folder file path to GazeFollow training and testing image annotations.")
        print("Download from GazeFollow Dataset authors. Example: '../data'.")
        print(args.data_folder_path)
        sys.exit()

    if args.method not in ['neural_net', 'softmax', 'svm']:
        print("Invalid method name. Try neural_net, softmax, or svm.")
        sys.exit()

    if args.method == 'svm' and \
        (len(args.svm_model_file_path) == 0 or not os.path.exists(args.svm_model_file_path)):
        print("Invalid file path to inception model.")
        print(svm_help_message)
        print("Example: '../../inception_dec_2015/tensorflow_inception_graph.pb'")
        sys.exit()

    training_file_path = os.path.join(args.data_folder_path, 'train_annotations.txt')
    testing_file_path = os.path.join(args.data_folder_path, 'test_annotations.txt')
	batchSize=25
	
    if args.method == 'neural_net':
        # training_dataset = Dataset(training_file_path, args.data_folder_path,
        #                            dataset_size=10000, batch_size=100,
        #                            width=28, height=28, depth=3)
        # testing_dataset = Dataset(testing_file_path, args.data_folder_path,
        #                           dataset_size=2000, batch_size=2000,
        #                           width=28, height=28, depth=3)
        training_dataset = Dataset(training_file_path, args.data_folder_path,
                                   dataset_size=constants.DATA_SIZE, batch_size=batchSize,
                                   width=28, height=28, depth=3)
        testing_dataset = Dataset(testing_file_path, args.data_folder_path,
                                  dataset_size=constants.DATA_SIZE, batch_size=batchSize,
                                  width=28, height=28, depth=3)
        nn_classifier(training_dataset, testing_dataset)

    elif args.method == 'softmax':
        # training_dataset = Dataset(training_file_path, args.data_folder_path,
        #                            dataset_size=5000, batch_size=100,
        #                            width=constants.IMAGE_WIDTH, height=constants.IMAGE_HEIGHT,
        #                            depth=constants.IMAGE_DEPTH)
        # testing_dataset = Dataset(testing_file_path, args.data_folder_path,
        #                           dataset_size=1000, batch_size=1000,
        #                           width=constants.IMAGE_WIDTH, height=constants.IMAGE_HEIGHT,
        #                           depth=constants.IMAGE_DEPTH)
        training_dataset = Dataset(training_file_path, args.data_folder_path,
                                   dataset_size=constants.DATA_SIZE, batch_size=batchSize,
                                   width=constants.IMAGE_WIDTH, height=constants.IMAGE_HEIGHT,
                                   depth=constants.IMAGE_DEPTH)
        testing_dataset = Dataset(testing_file_path, args.data_folder_path,
                                  dataset_size=constants.DATA_SIZE, batch_size=batchSize,
                                  width=constants.IMAGE_WIDTH, height=constants.IMAGE_HEIGHT,
                                  depth=constants.IMAGE_DEPTH)
        softmax_classifier(training_dataset, testing_dataset)

    elif args.method == 'svm':
        svm_classifier(training_file_path, args.data_folder_path, args.svm_model_file_path)

    # display images
    # annotations = annotation.image_annotations(image_file_path, data_file_path, dataset_size=10)
    # for annotation in annotations:
    #    display.image_with_annotation(annotation, GRID_SIZE)

if __name__ == '__main__':
    main()