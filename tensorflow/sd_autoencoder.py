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
    """Simple decorator that prints the execution time of a function"""
    @wraps(f)
    def wrapped(*args):
        start_time = time.time()
        result = f(*args)
        elapsed_time = time.time() - start_time
        print("Total time elapsed for execution of %s:" %str(f), elapsed_time)
        return result

    return wrapped

"""
#####################################
### STACKED DENOISING AUTOENCODER ###
#####################################
"""

class SDAutoencoder:
    """A stacked denoising autocoder implementation"""

    def __init__(self, dimensions):
        """
        dimensions (list): The number of neurons for each layer of
        the autoencoder. Ex: [784, 512, 256, 64]. The first item
        in the list should be the number of features in the input.
        The last item in the list should be the number of features
        in the output.
        """
        self.dimensions = dimensions
        self.x, \
        self.z, \
        self.y, \
        self.corrupt_prob, \
        self.cost = self._build()

    def _build(self):
        x = tf.placeholder(tf.float32, [None, self.dimensions[0]], name="x")

        # Probability of corrupting the input:
            # 1 for training
            # 0 for testing/production
        corrupt_prob = tf.placeholder(tf.float32, [1])
        current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

        # Build the encoder and the latent representation
        encoder, z = self._build_encoder(current_input)

        # Reconstructed input
        current_input = z
        y = self._decode(current_input, encoder)

        # Cost function measures pixel-wise difference
        cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))

        return x, z, y, corrupt_prob, cost

    def _build_encoder(self, input_tensor):
        """Builds the encoder based on a given input and `dimensions`

        input (Tensor): the input Tensor

        returns: ([Tensor], Tensor): a tuple containing:
            - A list of weight tensors used in each layer
            - The transformed input (z) after several layers
        """
        encoder = []
        for layer_i, n_output in enumerate(self.dimensions[1:]):
            n_input = int(input_tensor.get_shape()[1])
            W = tf.Variable(tf.random_uniform(
                shape=[n_input, n_output],
                minval=-1.0 / math.sqrt(n_input),
                maxval=1.0 / math.sqrt(n_input)
            ))
            b = tf.Variable(tf.zeros([n_output]))
            encoder.append(W)
            output = tf.nn.tanh(tf.matmul(input_tensor, W) + b)
            input_tensor = output

        return encoder, output

    def _decode(self, encoded_tensor, encoder):
        """Decodes the encoded_tensor by reversing operations using
        a reversed incoder

        encoded_tensor (Tensor): An encoded Tensor
        encoder ([Tensor]): The list of weight Tensors used to transform the original input

        returns: (Tensor) The decoded Tensor y
        """
        # Setup for building decoder
        encoder = encoder[::-1]

        # Build the decoder using the same weights
        for layer_i, n_output in enumerate(self.dimensions[:-1][::-1]):
            W = tf.transpose(encoder[layer_i])
            b = tf.Variable(tf.zeros([n_output]))
            output = tf.nn.tanh(tf.matmul(encoded_tensor, W) + b)
            encoded_tensor = output

        return output

@stopwatch
def test_mnist(dimensions):
    # Import and read MNIST data
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)

    # Create an SDA with dimensions.length() - 1 layers
    ae = SDAutoencoder(dimensions=dimensions)

    # Create an Adam optimizer for gradient descent
    learning_rate = 1e-4
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae.cost)

    # Setup accuracy model
    # is_correct = tf.equal()

    # Initialize the default session graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Set batch and epoch size
    batch_size = 50
    n_epochs = 10

    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            feed = {ae.x: train, ae.corrupt_prob: [1.0]}
            sess.run(optimizer, feed_dict=feed)

            if batch_i % 100 == 0:
                pass
        print(epoch_i, sess.run(ae.cost, feed_dict=feed))

    # Plotting
    n_examples = 15
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae.y, feed_dict={
        ae.x: test_xs_norm, ae.corrupt_prob: [0.0]})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape([recon[example_i, :] + mean_img], (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()

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

def main():
    s = SDAutoencoder([784, 256, 128, 64])
    test_mnist([28 * 28, 512, 256, 64])

if __name__ == "__main__":
    main()
