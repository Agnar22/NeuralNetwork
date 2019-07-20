# NeuralNetwork
This is a python implementation of a feed-forward neural network using numpy.
The network is trained by mini-batch graident descent and uses l2 regularization.

## Installation
Clone the repository
```bash
git clone https://github.com/Agnar22/NeuralNetwork.git
```

navigate into the project folder
```bash
cd NeuralNetwork
```

install requirements
```bash
pip install -r requirements.txt
```

if everything went well you should now be able to run the code
```bash
python3 Main.py
```

## Motivation
I created this project to get insight into the mathematics behind backpropagation in neural networks, 
as well as learning how to implement it by only using matrix operations. Numpy is used for the matrix operations.

To check if the neural network (both feed forward and backpropagation) was working, I tested it on the MNIST dataset (supplied py tensorflow).

## Results

<p align='center'>
<img width="50%" src="https://github.com/Agnar22/NeuralNetwork/blob/master/README_images/graph.PNG"><br>
<b>Figure 1</b>: the training- and validation loss for each epoch
</p>
The loss for the training- (in blue) and validation data (in yellow) is shown above (<b>Figure 1</b>). It is strictly decreasing, like it should be. The regression loss used was l2 and the classifiacation loss used was quadratic loss.<br><br>

<p align='center'>
<img width="50%" src="https://github.com/Agnar22/NeuralNetwork/blob/master/README_images/predictions.PNG"><br>
<b>Figure 2</b>: a matrix showing target (rows) and prediction (columns) by the NN for the validation data
</p>
By looking at the prediction matrix for the validation data (<b>Figure 2</b>), you can see that the network easily recognizes 0's and 1's (with an accuracy of 96% and 97% respectively). On the other hand, 7's and 8's proved to be more difficult. The former where often misclassified as 9 whereas the latter where frequently misclassified as 3 and 5. Both of the misclassifications are understandable; for a neural network that does not consider spatial invariances, like a conv-net, a sloppy handwritten 7 might resemble a 9, and the curves of a 8 might look like the curves of a 3 or a 5.<br><br>


<p align='center'>
<img width="80%" src="https://github.com/Agnar22/NeuralNetwork/blob/master/README_images/statistics.PNG"><br>
<b>Figure 3</b>: training- and validation loss and accuracy for each epoch
</p>
With a mini-batch size of 64 and 5 epochs, the neural network managed to get a final validation accuracy of 88.97% (<b>Figure 3</b>).
Compared to todays conv-nets, which get staggeringly close to a 100% accuracy, this is not impressive. One could argue that further hyperparameter tuning would improve the result a few percent. However, taking into account that the objective of this project was to understand the underlaying maths behinde backpropagation and being able to implement it, I would say that 88.97% is more than sufficient to prove that the implementation is correct.


## Other resources
* [This book](http://neuralnetworksanddeeplearning.com/index.html "Neural networks and deep learning") about neural networks.
* [This](https://towardsdatascience.com/a-step-by-step-implementation-of-gradient-descent-and-backpropagation-d58bda486110 "A step by step implementation of gradient descent and backpropagation") short explanation- and implementation of backpropagation from towardsdatascience.
* Figure 1 from [this paper](https://www.researchgate.net/publication/277411157_Deep_Learning/link/55e0cdf908ae2fac471ccf0f/download "Deep learning paper by Yann LeCun et al.") showing how the gradient is calculated for each layer.

## License
This project is licensed under the MIT License.