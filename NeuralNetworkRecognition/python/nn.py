import numpy as np
from util import *

# do not include any more libraries here!
# do not put any code outside of functions!


############################## Q 2.1.2 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b.
# X be [Examples, Dimensions]
def initialize_weights(in_size, out_size, params, name=""):
    W, b = None, None

    #define limit    
    limit = (6**0.5) / np.sqrt(in_size + out_size)
    
    #initialize weights and bias
    W = np.random.uniform(-limit, limit, (in_size, out_size))
    b = np.zeros((out_size))

    #output to params dictionary
    params["W" + name] = W
    params["b" + name] = b


############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):

    res = 1.0 / (1.0 + np.exp(-x))
    return res

############################## Q 2.2.1 ##############################
def forward(X, params, name="", activation=sigmoid):
    
    pre_act, post_act = None, None
    # get the layer parameters
    W = params["W" + name]
    b = params["b" + name]

    # your code here
    pre_act = X @ W + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params["cache_" + name] = (X, pre_act, post_act)

    return post_act


############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):

    c = np.max(x, axis=1, keepdims=True) # find max value

    res = np.exp(x-c) / np.sum(np.exp(x-c), axis=1, keepdims=True) # softmax function  
    return res


############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    mask = y.astype(bool) #given y, only select probabilities associated with true classes
    probsTrue = probs[mask] # get the true probabilities
    loss = 0.0 - np.sum(np.log(probsTrue)) #apply negative log loss function
    
    classPred = np.argmax(probs, axis=1) # get the predicted class
    predCorrect = mask[np.arange(classPred.shape[0]), classPred] #check how many predictions are correct by comparing true class with predicted class

    count = np.sum(predCorrect) #count the number of correct predictions

    acc = float(count) / y.shape[0] #calculate accuracy

    return loss, acc


############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act * (1.0 - post_act)
    return res


def backwards(delta, params, name="", activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params["W" + name]
    b = params["b" + name]
    X, pre_act, post_act = params["cache_" + name]

    delta_pre = activation_deriv(post_act) * delta # calculate delta pre
    #calculate gradients
    grad_W = X.transpose() @ delta_pre
    grad_b = np.sum(delta_pre, axis=0)
    grad_X = delta_pre @ W.transpose()

    # store the gradients
    params["grad_W" + name] = grad_W
    params["grad_b" + name] = grad_b
    return grad_X


############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x, y, batch_size):
    batches = []

    N = x.shape[0]

    assert batch_size <= N
    #randomize indices
    indices = list(np.random.permutation(N))
    #append on indices to make sure all examples are used
    remainer_num = N % batch_size
    if remainer_num > 0:
        append_on_num = batch_size - remainer_num
        for i in range(append_on_num):
            indices.append(i)

    #split into batches
    p = 0
    while p < N:
        batch_idxs = indices[p:p+batch_size]
        
        batches.append((x[batch_idxs, :], y[batch_idxs, :]))

        p += batch_size

    return batches
