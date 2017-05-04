# GazeFollow

Setup instructions:

For MacOS:
virtualenv --system-site-packages -p python3 GazeFollow
source bin/activate

Libraries:
scipy
numpy
PIL, pillow
tensorflow
matplotlib

Results:
Neural net classifier
Euclidean dist: 0.400344 with combined cross entropy
Euclidean dist: 0.390389
Accuracy 0.058, time to completion ~5 min.
training_dataset_size = 10000, training_batch_size = 100
testing_dataset_size = 2000, testing_batch_size = 2000

Softmax classifier 
Accuracy 0.24, euclidean: 0.32
training_dataset_size = 100
training_batch_size = 10
testing_dataset_size = 50
testing_batch_size = 50

Accuracy 0.05, euclidean: .52
training_dataset_size = 10000
training_batch_size = 100
testing_dataset_size = 2000
testing_batch_size = 2000

combined euc dist. 0.415177
training_dataset_size = 5000
training_batch_size = 100
testing_dataset_size = 1000
testing_batch_size = 1000

Selected quotes from "Where are they looking", Recasens et al.

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