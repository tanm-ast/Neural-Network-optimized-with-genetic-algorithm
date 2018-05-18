# Neural-Network-optimized-with-genetic-algorithm
This is a simple project which involves a neural network trained for the purpose of face detection Extended Yale B database
(a collection of faces of 38 persons). Histogram of Oriented Gradients (HoG) features have been extracted and stored in a separate
.mat file which is loaded along with labels file while training and testing the neural network. The code can be easily edited
so as to cater to any other kind of data set. The neural network is coded from scratch without using any in-built functions as
this was a part of my course project for Pattern Recognition course. However, for HoG in-built matlab functions are used.

k-fold cross validation is used in training.
 
The weights of the neural network are further optimized using RMS propagation algorithm. Other methods that can be easily 
incorporated are Adadelta, Adam, Adagrad, Nesterov momentum et al.

The activation function choices available as of now are Sigmoid and Tanh. The cost-function used is minimum mean square.
Minibatch steepest gradient descent along with backpropagation is used to update the weights and biases of the network.

In the main-code a population of members is initialized with different numbers of hidden layers and neurons in each hidden 
layer. The memembers are then trained using the training data-set and average accuracy is calculated using k-fold cross
validation. The accuracy is used as fitness test for selecting the fittest members from the population.

For the next generation a certain number of fittest members and some random lucky suvivors are selected and these members 
are bred using cross-linking to generate children for new generation. Some children also have their contents mutated, this
being dependent on a mutation probability.

As of now the project invlolves only training and no explicit testing. Coding is done in matlab.

Future edits may include 
1) Incorporating various weight optimization techniques mentioned above.
2) More types of activation & cost functions.
3) rying training with more varied datasets and features.

References: 

1) http://cs231n.github.io/
2) https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
3) http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html
4) http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html
5) https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
6) https://www.mathworks.com/help/vision/ref/extracthogfeatures.html
