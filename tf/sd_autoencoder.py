"""
Stacked Denoising Autoencoder implementation

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


class SoftmaxLayer(object):

    def __init__(self, input_tensor, n_input, n_output):
        """A logistic regression layer described by a weight matrix W and a bias b.

        :param input_tensor: The input tensor.
        :param n_input: The dimension of the input.
        :param n_output: The dimension of the output labels.
        """

        self.W = tf.Variable(tf.zeros([n_input, n_output], tf.float32), name="W")
        self.b = tf.Variable(tf.zeros([n_output]), name="b")

        self.p_y_given_x = tf.nn.softmax(tf.matmul(input_tensor, self.W) + self.b)

        self.y_pred = tf.argmax(self.p_y_given_x, 1)
        self.params = [self.W, self.b]
        self.input = input_tensor


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
            ), name="W")
            W = W * 4 if act == tf.nn.sigmoid else W

        if b is None:
            b = tf.Variable(tf.zeros([n_output]), name="b")

        self.W = W
        self.b = b

        linear_output = tf.matmul(input_tensor, self.W) + self.b
        self.output = linear_output if act is None else act(linear_output)

        self.params = [self.W, self.b]


class DAutoencoder(object):

    def __init__(self, n_visible, n_hidden, input_tensor=None, W=None,
                 bhid=None, bvis=None, act=tf.nn.sigmoid):
        """Initializes a denoising autoencoder by specifying the input dimension
        and the dimension of the hidden layer. If using an SDA, inputs and weights
        are the outputs of the previous layer.

        :param n_visible: The input dimension.
        :param n_hidden: The dimension of the hidden layer.
        :param input_tensor: An optional input tensor variable.
        :param W: An optional weight tensor. If the DAutoencoder is supposed to be
            standalone, then this should be None.
        :param bhid: An optional bias tensor for hidden units. If the DAutoencoder
            is supposed to be standalone, then this should be None.
        :param bvis: An optional bias tensor for visible units. If the DAutoencoder
            is supposed to be standalone, then this should be None.
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not W:
            W = tf.Variable(tf.random_uniform(
                shape=[n_visible, n_hidden],
                minval=-np.sqrt(6. / (n_visible + n_hidden)),
                maxval=np.sqrt(6. / (n_visible + n_hidden))
            ), name="W")

        if not bvis:
            bvis = tf.Variable(tf.zeros([n_visible]))

        if not bhid:
            bhid = tf.Variable(tf.zeros([n_hidden]), name="b")

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = tf.transpose(self.W)
        self.act = act

        self.x = input_tensor if input_tensor is not None else tf.placeholder(tf.float32)

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input_tensor, corruption_level): #FIXME: Bugged multiplication: must be only 1s
        """Keeps `1-corruption_level` proportion of entries of inputs, and zeros
        a random subset of proportion `corruption_level` from the input.

        :param input_tensor:
        :param corruption_level: Between 0 and 1, the proportion of input to corrupt.
        :return: The corrupted input.
        """

        corruption = tf.cast(tf.random_uniform(
            shape=tf.shape(input_tensor),
            minval=0,
            maxval=int(1 / corruption_level),
            dtype=tf.int32
        ), tf.float32)

        return tf.mul(input_tensor, corruption)

    def get_hidden_values(self, input_tensor):
        """Computes the values of the hidden layer.

        :param input_tensor:
        :return:
        """

        return self.act(tf.matmul(input_tensor, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given values of hidden layer.

        :param hidden:
        :return:
        """

        return self.act(tf.matmul(hidden, self.W_prime), + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """Computes the cost and the updates for training one step of the denoising autoencoder.

        :param corruption_level:
        :param learning_rate:
        :return:
        """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(
            self.x * tf.log(z) + (1 - self.x) * tf.log(1 - z), reduction_indices=[1]))

        gparams = tf.gradients(cross_entropy, self.params)
        updates = [ #FIXME Not sure if this is the same implementation in TF
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return cross_entropy, updates


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
        self.da_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0, "Error: Number of layers must be greater than zero."

        self.x = tf.placeholder(tf.float32, [None, n_input], name="x")
        self.y = tf.placeholder(tf.float32, [None, n_output], name="y")

        for i in range(self.n_layers):
            input_dim = n_input if i == 0 else hidden_layers_sizes[i - 1]
            layer_input = self.x if i == 0 else self.act_layers[-1].output

            act_layer = HiddenLayer(layer_input, input_dim, hidden_layers_sizes[i], act=act)
            self.act_layers.append(act_layer)
            self.params.extend(act_layer.params)

            da_layer = DAutoencoder(n_visible=input_dim,
                                    n_hidden=hidden_layers_sizes[i],
                                    input_tensor=layer_input,
                                    W=act_layer.W,
                                    bhid=act_layer.b)

            self.da_layers.append(da_layer)

        #FIXME Add softmax regression layer on top
        self.log_layer = SoftmaxLayer(input=self.act_layers[-1].output,
                                      n_input=hidden_layers_sizes[-1],
                                      n_output=n_output)
        self.params.extend(self.log_layer.params)

        self.finetune_cost = self.log_layer