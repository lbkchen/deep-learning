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
######################
### HELPER CLASSES ###
######################
"""

class HiddenLayer(object):
    def __init__(self, input_tensor, n_input, n_output, W=None, b=None, act=tf.nn.tanh):
        """A hidden layer of a multi-layer perceptron. Units are fully connected
        and have activation function act. Weight matrix W is of shape
        [n_input, n_output] and the bias vector has shape [n_out].

        :param input_tensor: The tensor input.
        :param n_input: The dimension of the input.
        :param n_output: The dimension of the output.
        :param W: The weight matrix.
        :param b: The bias vector.
        :param act: The activation function of the neurons.
        """

        self.input = input_tensor

        if W is None:
            W = tf.Variable(tf.random_uniform(
                shape=[n_input, n_output],
                minval=-np.sqrt(6. / (n_input + n_output)),
                maxval=np.sqrt(6. / (n_input + n_output))
            ))
            W = W * 4 if act == tf.nn.sigmoid else W

        if b is None:
            b = tf.Variable(tf.zeros([n_output]))

        self.W = W
        self.b = b

        linear_output = tf.matmul(input_tensor, self.W) + self.b
        self.output = linear_output if act is None else act(linear_output)

        self.params = [self.W, self.b]

"""
#####################################
### STACKED DENOISING AUTOENCODER ###
#####################################
"""

class SDAutoencoder(object):
    """A stacked denoising autocoder implementation."""

    def __init__(self, n_input, n_output, hidden_layers_sizes, corruption_levels, act=tf.nn.tanh):
        """Creates a Stacked Denoising Autoencoder instance with a variable amount
        of layers.

        :param n_input: The dimension of the input layer.
        :param n_output: The dimension of the output layer.
        :param hidden_layers_sizes: A list of at least one value. The dimensions
            of all intermediate layers.
        :param corruption_levels: Corresponds to hidden_layer_sizes, amount of
            corruption used for each hidden layer.
        :param act: The activation function used for the neurons in each layer.
        """

        self.act_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0, "Error: Number of layers must be greater than zero."

        self.x = tf.placeholder(tf.float32, [None, n_input], name="x")
        self.y = tf.placeholder(tf.float32, [None, n_output], name="y")

        for i in range(self.n_layers):
            input_dim = n_input if i == 0 else hidden_layers_sizes[i - 1]
            layer_input = self.x if i == 0 else self.act_layers[-1].output

            layer = HiddenLayer(layer_input, input_dim, hidden_layers_sizes[i], act=act)
            self.act_layers.append(layer)
            self.params.extend(layer.params)

            # dA_layer = dA FIXME: Need to implement denoising autoencoder