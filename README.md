# GazeFollow

Setup instructions:
    `pip3 install setup.py`

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

Accuracy 0.05, euclidean: .52
training: dataset_size = 10000, batch_size = 100
testing: dataset_size = 2000, batch_size = 2000

combined euc dist. 0.415177
training: dataset_size = 5000, batch_size = 100
testing: dataset_size = 1000, batch_size = 1000

SVM Classifier:
Euclidean distance: 0.45
1000 images

Reference:
Recasens et al, "Where are they looking?"