# GazeFollow

Setup instructions:
    Download GazeFollow dataset from `http://gazefollow.csail.mit.edu/`
    `pip3 install setup.py`

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
    scipy
    numpy
    pillow
    tensorflow
    sklearn
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

Reference:
Recasens et al, "Where are they looking?"