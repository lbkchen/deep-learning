"""
Denoising Autoencoders example with MNIST

Ken Chen
"""

import tensorflow as tf
import numpy as np
import math
import time
from functools import wraps

"""
##################
### DECORATORS ###
##################
"""

def stopwatch(f):
    """Simple decorator that prints the execution time of a function."""
    @wraps(f)
    def wrapped(*args):
        start_time = time.time()
        result = f(*args)
        elapsed_time = time.time() - start_time
        print("Total time elapsed for execution of %s:" %f, elapsed_time)
        return result

    return wrapped

"""
#####################################
### STACKED DENOISING AUTOENCODER ###
#####################################
"""

class SDAutoencoder(object):
    """A stacked denoising autocoder implementation."""

    def __init__(self, n_input, n_output, hidden_layers_sizes, corruption_levels):
        """Creates a Stacked Denoising Autoencoder instance with a variable amount
        of layers.

        :param n_input: The dimension of the input layer.
        :param n_output: The dimension of the output layer.
        :param hidden_layers_sizes: A list of at least one value. The dimensions
            of all intermediate layers.
        :param corruption_levels: Corresponds to hidden_layer_sizes, amount of
            corruption used for each hidden layer.
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0, "Error: Number of layers must be greater than zero."

        self.x = tf.placeholder(tf.float32, [None, n_input], name="x")
        self.y = tf.placeholder(tf.float32, [None, n_output], name="y")

        for i in range(self.n_layers):
            input_dim = n_input if i == 0 else hidden_layers_sizes[i - 1]
            layer_input = self.x if i == 0 else self.sigmoid_layers[-1].output
