"""
Denoising Autoencoders

Ken Chen
"""

import tensorflow as tf
import numpy as np
import math

class SDAutoencoder:

    def __init__(dimensions):
        """
        dimensions (list): The number of neurons for each layer of the autoencoder. Ex: [784, 512, 256, 64]. The first item in the list
        should be the number of features in the input. The last item in
        the list should be the number of features in the output.
        """
        self.dimensions = dimensions

    def _build():
        x = tf.placeholder(tf.float32, [None, dimensions[0]], name="x")

        corrupt_prob = tf.placeholder(tf.float32, [1])
        current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

        # Build the encoder
        encoder = []
        for layer_i, n_output in enumerate(dimensions[1:]):
            n_input = int(current_input.get_shape()[1])
            W = tf.Variable(tf.random_uniform(
                shape=[n_input, n_output],
                minval=-1.0 / math.sqrt(n_input),
                maxval=1.0 / math.sqrt(n_input)
            ))
            b = tf.Variable(tf.zeros([n_output]))
            encoder.append(W)
            output = tf.nn.tanh(tf.matmul(current_input, W) + b)
            current_input = output

        # Latent representation:: input after several layers of
        # transformation
        z = current_input
        encoder.reverse()

    """
    ########################
    ### HELPER FUNCTIONS ###
    ########################
    """
    def corrupt(x):
        """Takes an input tensor and corrupts half of the values uniformly by zeroing them.

        x (Tensor): input to corrupt
        returns: (Tensor) input with 50 percent of values corrupted
        """
        corruption = tf.cast(tf.random_uniform(
            shape=tf.shape(x),
            minval=0,
            maxval=2,
            dtype=tf.int32
        ), tf.float32)

        return tf.mul(x, corruption)
