"""
Stacked Denoising Autoencoder Implementation

Ken Chen
"""

"""
###########################
### SETUP AND VARIABLES ###
###########################
"""


import tensorflow as tf
import numpy as np
import time
import csv
from functools import wraps
from sklearn.preprocessing import MaxAbsScaler


allowed_activations = ["sigmoid", "tanh", "relu", "softmax"]
allowed_losses = ["rmse", "cross-entropy"]

xs_filepath = "../data/S01X.csv"
ys_filepath = "../data/S01Y.csv"


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
        print("Total seconds elapsed for execution of %s:" %f, elapsed_time)
        return result

    return wrapped


"""
##################################
### HELPER CLASSES / FUNCTIONS ###
##################################
"""


def get_next_batch(filename, batch_size):
    """Generator that gets the net batch of batch_size x or y values
    from the given file.

    :param filename:
    :param criterion:
    :return:
    """
    with open(filename, "rt") as file:
        reader = csv.reader(file)
        index = 0
        max_index = len(reader) // 2
        this_batch = []
        for row in reader:
            this_batch.append(row)
            index += 1

            if index > max_index:
                break

            if index % batch_size == 0:
                yield this_batch
                this_batch = []


"""
#####################################
### STACKED DENOISING AUTOENCODER ###
#####################################
"""


class SDAutoencoder:
    """A stacked denoising autoencoder."""

    def check_assertions(self):
        global allowed_activations, allowed_losses
        assert self.loss in allowed_losses, "Incorrect loss given."
        assert 'list' in str(type(self.dims)), "dims must be a list even if there is one layer."
        assert len(self.epochs) == len(self.dims), "No. of epochs must equal to no. of hidden layers."
        assert len(self.activations) == len(self.dims), "No. of activations must equal to no. of hidden layers."
        assert all(True if x > 0 else False for x in self.epochs), "No. of epoch must be at least 1."
        assert set(self.activations + allowed_activations) == set(allowed_activations), "Incorrect activation given."
        assert 0 <= self.noise <= 1, "Invalid noise value given: %s" % self.noise

    def __init__(self, dims, activations, epochs, noise=0, loss="cross-entropy",
                 lr=0.0001, batch_size=100, print_step=50):
        """

        :param dims:
        :param activations:
        :param epochs:
        :param noise:
        :param loss:
        :param lr:
        :param batch_size:
        :param print_step:
        """
        self.dims = dims
        self.activations = activations
        self.epochs = epochs
        self.noise = noise
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size
        self.print_step = print_step

        self.check_assertions()
        self.depth = len(dims)
        self.weights = []
        self.biases = []

    def corrupt(self, tensor, corruption_level=0.5):
        """Uses the masking noise algorithm to mask corruption_level proportion
        of the input."""
        # FIXME: Currently only corrupts 50% regardless
        corruption = tf.cast(tf.random_uniform(
            shape=tf.shape(tensor),
            minval=0,
            maxval=2,
            dtype=tf.int32
        ), tf.float32)

        return tf.mul(tensor, corruption)

    def train_layer(self, input_dim, output_dim, num_batches, act=tf.nn.sigmoid):
        sess = tf.Session()

        x_true = tf.placeholder(tf.float32, shape=[None, input_dim])
        x_corrupt = self.corrupt(x_true)

        encode = {"weights": tf.Variable(tf.truncated_normal([input_dim, output_dim], dtype=tf.float32)),
                  "biases": tf.Variable(tf.truncated_normal([output_dim], dtype=tf.float32))}

        decode = {"weights": tf.transpose(encode["weights"]),
                  "biases:": tf.Variable(tf.truncated_normal([input_dim], dtype=tf.float32))}

        encoded = act(tf.matmul(x_corrupt, encode["weights"]) + encode["biases"])
        decoded = tf.matmul(encoded, decode["weights"]) + decode["biases"]

        # Reconstruction loss
        loss = self.get_loss(x_true, decoded)

        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        sess.run(tf.initialize_all_variables())

        for i in range(num_batches):
            batch_x_true =

    def get_loss(self, tensor_1, tensor_2):
        if self.loss == "rmse":
            return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(tensor_1, tensor_2))))
        elif self.loss == "cross-entropy":
            return tf.reduce_mean(-tf.reduce_sum(
                tensor_1 * tf.log(tensor_2) + (1 - tensor_1) * tf.log(1 - tensor_2), reduction_indices=[1]
            ))