# GazeFollow

This project implements several computer vision machine learning methods for 
locating where subjects in images are looking.  
It was inspired by the paper "Where are they looking?" which can be found at 
`http://gazefollow.csail.mit.edu/`.
To see report giving full results check Report directory
The dataset will need to be downloaded from there, it is 6 gigs big so we can't
upload it via canvas.
In addition for SVM a pretrained model is needed we got ours from
https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip

For the matlab version
Run main.m in the matlab code file. It assumes that training images are in a train directory, testing images
are in a test directory and train_annotations & test_annotations from the gazefollow project are in the local directory.

For the python project using tensorflow
Installation instructions:
    Download GazeFollow dataset from `http://gazefollow.csail.mit.edu/`
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

Libraries:
    scipy, numpy, pillow
    tensorflow, sklearn
    matplotlib

Results:
    Neural net classifier
    Euclidean dist: 0.400344 with combined cross entropy
    Euclidean dist: 0.390389
    Accuracy 0.058, time to completion ~5 min.
    training: dataset_size = 10000, batch_size = 100
    testing: dataset_size = 2000, batch_size = 2000

    Softmax classifier 
    Accuracy 0.24, euclidean: 0.32
    training: dataset_size = 100, batch_size = 10
    testing: dataset_size = 50, batch_size = 50

    combined euc dist. 0.415177
    training: dataset_size = 5000, batch_size = 100
    testing: dataset_size = 1000, batch_size = 1000

    SVM Classifier:
    Euclidean distance: 0.45
    training: 800
    testing: 200

GazeFollow Reference:
A. Recasens*, A. Khosla*, C. Vondrick and A. Torralba, 
"Where are they looking?",
Advances in Neural Information Processing Systems (NIPS), 2015.
(* - indicates equal contribution)
http://gazefollow.csail.mit.edu/