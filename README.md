# neural-nets-in-numpy
This is a storage place for practice code that implements parts of neural network architecture and training methods in numpy. 
# classes:
## denseLayer
this is the basic layer class, which has attributes 
* `bias`: a column vector, stored as a 2-dimensional numpy array,
* `linear`: a matrix, stored as a 2-dimensional numpy array,
* `activation`: a string or None. So far the reLU, sigmoid, and tanh functions have been implemented, as well as softmax if it appears at the end of a neural network.

It also supports the following methods
* `linearPass`: by default sends a column vector v to "`bias` + `linear`@ v". There is an optional boolean argument addBias, which can be set to False to just send v to "`linear` @ v".
* `forwardPass`: sends v to the output of linearPass passed through the activation function corresponding to `activation`,
* `activationDerDiag`: viewing `activation` as a map from R^n to R^n that sends (x_i) to f(x_i), this takes an input vector (y_i) represented as a 2-dimensional numpy array, and returns a 1-dimensional numpy array (f'(y_i)). This gets used in updating the gradient during backpropagation. However, if `activation` == `softmax` this actually just returns a sequence of 1s which is because as it stands now

## nNet
this is the neural network class, which has the attribute
* `layersList`: a list of layers which are executed sequentially. 

It has methods
* `forwardPass`: by default this takes as input a column vector and returns the outputs after each layer (keeping these outputs is intended to save time during backpropagation). If an optional boolean argument is set to False, it returns only the final output.
* `backprop`: this takes as input 
  * `data`, a list of pairs of column vectors, 
  * `loss`, a string. Right now, `mse` (mean-squared error) and `cel` (cross entropy loss) are implemented. For `cel` to work as intended, the last layer in the neural network has to be a softmax, 
  * `learning_rate`, some float,
  
  and updates the `bias` and `linear` terms in each layer using stochastic gradient descent.
  
* `nEpochs`: which takes the same arguments as backprop as well as an integer `n`, and iterates backprop `n` times.
