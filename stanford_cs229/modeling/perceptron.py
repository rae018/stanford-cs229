import math

import matplotlib.pyplot as plt
import numpy as np


def initial_state(dim):
    """Return the initial state for the perceptron.

    This function computes and then returns the initial state of the perceptron.
    Feel free to use any data type (dicts, lists, tuples, or custom classes) to
    contain the state of the perceptron. The state of the perceptron can include
    anything the model must store to make predictions.
    """
    return {'betas' : [0], 'xs' : [np.zeros(dim)]}


def update_state(state, kernel, learning_rate, x_i, y_i):
    """Updates the state of the perceptron.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns the result of a kernel
        learning_rate: The learning rate for the update
        x_i: A vector containing the features for a single instance
        y_i: A 0 or 1 indicating the label for a single instance
    """
    betas = state['betas']
    xs = state['xs']

    sign_arg = 0
    for j in range(len(betas)):
        sign_arg += betas[j] * kernel(xs[j], x_i)
        
    betas += [learning_rate * (y_i - sign(sign_arg))]
    xs += [x_i]


def sign(a):
    """Gets the sign of a scalar input."""
    if a >= 0:
        return 1
    else:
        return 0


def dot_kernel(a, b):
    """An implementation of a dot product kernel.

    Args:
        a: A vector
        b: A vector
    """
    return np.dot(a, b)


def rbf_kernel(a, b, sigma=0.1):
    """An implementation of the radial basis function kernel.

    Args:
        a: A vector
        b: A vector
        sigma: The radius of the kernel
    """
    distance = (a - b).dot(a - b)
    scaled_distance = -distance / (2 * (sigma) ** 2)
    return math.exp(scaled_distance)

def non_psd_kernel(a, b):
    """An implementation of a non-psd kernel.

    Args:
        a: A vector
        b: A vector
    """
    if(np.allclose(a,b,rtol=1e-5)):
        return -1
    return 0

def train_perceptron(X, Y, kernel=rbf_kernel, learning_rate=0.5):
    """Train a perceptron with the given kernel.

    This function trains a perceptron with a given kernel and then
    uses that perceptron to make predictions.
    The output predictions are saved to src/perceptron/perceptron_{kernel_name}_predictions.txt.
    The output plots are saved to src/perceptron/perceptron_{kernel_name}_output.pdf.

    Args:
        kernel: The kernel function.
        learning_rate: The learning rate for training.
    """
    _, dim = X.shape
    state = initial_state(dim)
    
    iteration = 0
    for x_i, y_i in zip(X, Y):
        if iteration % 100 == 0:
          print('Completed {} iterations'.format(iteration))
        update_state(state, kernel, learning_rate, x_i, y_i)
        iteration += 1
        
    return state
  
def predict_perceptron(state, x_i, kernel=rbf_kernel):
    """Peform a prediction on a given instance x_i given the current state
    and the kernel.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns
            the result of a kernel
        x_i: A vector containing the features for a single instance

    Returns:
        Returns the prediction (i.e 0 or 1)
    """
    betas = state['betas']
    xs = state['xs']

    result = 0
    for j in range(len(betas)):
        result += betas[j] * kernel(xs[j], x_i)
    
    return sign(result)
