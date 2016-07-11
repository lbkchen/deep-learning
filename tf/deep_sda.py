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
###########################
### SETUP AND VARIABLES ###
###########################
"""


allowed_activations = ["sigmoid", "tanh", "relu", "softmax"]
allowed_noises = [None, "gaussian", "mask"]
allowed_losses = ["rmse", "cross-entropy"]


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


"""
#####################################
### STACKED DENOISING AUTOENCODER ###
#####################################
"""


class SDAutoencoder:
    """A stacked denoising autocoder implementation."""

    def check_assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses, "Incorrect loss given."
        assert 'list' in str(type(self.dims)), "dims must be a list even if there is one layer."
        assert len(self.epoch) == len(self.dims), "No. of epochs must equal to no. of hidden layers."
        assert len(self.activations) == len(self.dims), "No. of activations must equal to no. of hidden layers."
        assert all(True if x > 0 else False for x in self.epoch), "No. of epoch must be at least 1."
        assert set(self.activations + allowed_activations) == set(allowed_activations), "Incorrect activation given."
        assert self.noise in allowed_noises, "Incorrect noise given."

    def __init__(self, dims, activations, epoch=1000, noise=None, loss="cross-entropy",
                 lr=0.0001, batch_size=100, print_step=50):
        """Creates the autoencoder

        :param dims:
        :param activations:
        :param epoch:
        :param noise:
        :param loss:
        :param lr:
        :param batch_size:
        :param print_step:
        """
        self.dims = dims
        self.activations = activations
        self.epoch = epoch
        self.noise = noise
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size
        self.print_step = print_step
        self.check_assertions()
        self.depth = len(dims)
        self.weights = []
        self.biases = []

    def add_noise(self, x):
        if self.noise == "gaussian":
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if "mask" in self.noise:
            frac = float(self.noise.split("-")[1]) #FIXME Dont really get this
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), round(frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == "sp":
            pass

    def fit(self, x):
        for i in range(self.depth):
            print("Layer %d" %(i + 1))
            if self.noise is None:
                x = self.run(data_x=x, activation=self.activations[i], data_x_=x,
                             hidden_dim=self.dims[i], epoch=self.epoch[i], loss=self.loss,
                             batch_size=self.batch_size, lr=self.lr, print_step=self.print_step)
            else:
                x = self.run(data_x=self.add_noise(x), activation=self.activations[i], data_x_=x,
                             hidden_dim=self.dims[i], epoch=self.epoch[i], loss=self.loss,
                             batch_size=self.batch_size, lr=self.lr, print_step=self.print_step)
        print(x)

    def transform(self, data):
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)
        return x.eval(session=sess)