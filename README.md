# CS 6001: Computer Vision Project

This project implements several computer vision machine learning methods for 
locating where subjects in images are looking.

It was inspired by the paper "Where are they looking?" which can be found at 
`http://gazefollow.csail.mit.edu/`.

The image datasets will need to be downloaded from there.

For the matlab version, run main.m in the matlab code file. It assumes that training images are in a train directory, testing images
are in a test directory and train_annotations & test_annotations from the gazefollow project are in the local directory.

For the python project using tensorflow
Installation instructions:
    Download GazeFollow dataset from `http://gazefollow.csail.mit.edu/`
    To run SVM a pretrained model is needed.  Download from:
        https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
    `pip3 install setup.py` if that doesn't work please do 
	'python setup.py develop'

For usage instructions:
    `gazefollow -h`

The following usage examples assume the GazeFollow dataset is located in a folder called `data`.

Run Neural Net Classifier:
    Example: `gazefollow data`

Run Softmax Classifier:
    Example: `gazefollow data -m softmax`

Run SVM Classifier:
    Assuming the inception model is located at: 
        `../inception_dec_2015/tensorflow_inception_graph.pb`
        
    Example: `gazefollow data -m svm -s ../inception_dec_2015/tensorflow_inception_graph.pb`

GazeFollow Reference:
A. Recasens*, A. Khosla*, C. Vondrick and A. Torralba, 
"Where are they looking?",
Advances in Neural Information Processing Systems (NIPS), 2015.
(* - indicates equal contribution)
http://gazefollow.csail.mit.edu/