"""
Stacked Denoising Autoencoder Implementation

Ken Chen
"""

"""
###########################
### SETUP AND CONSTANTS ###
###########################
"""


import tensorflow as tf
import numpy as np
import time
import csv
from functools import wraps


ALLOWED_ACTIVATIONS = ["sigmoid", "tanh", "relu", "softmax"]
ALLOWED_LOSSES = ["rmse", "cross-entropy"]

X_TRAIN_PATH = "../data/splits/PXTrainSAM.csv"
Y_TRAIN_PATH = "../data/splits/PYTrainSAM.csv"
X_TEST_PATH = "../data/splits/PXTestSAM.csv"
Y_TEST_PATH = "../data/splits/YTestSAM.csv"


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


def get_next_batch(filename, batch_size, skip_header=True):
    """Generator that gets the net batch of batch_size x or y values
    from the given file.

    :param filename: A string of the file to write to.
    :param batch_size: The number
    :return:
    """
    with open(filename, "rt") as file:
        reader = csv.reader(file)

        if skip_header:
            next(reader)

        index = 0
        this_batch = []  # FIXME: Can probably optimize to take numpy array
        for row in reader:
            this_batch.append(row)
            index += 1

            if index % batch_size == 0:
                yield this_batch
                this_batch = []


"""
#####################################
### STACKED DENOISING AUTOENCODER ###
#####################################
"""


class NNLayer:
    """A container class to represent a hidden layer in the autoencoder network."""

    def __init__(self, input_dim, output_dim, activation=lambda x: x, weights=None, biases=None):
        """Initializes an NNLayer with empty weights/biases (default). Weights/biases
        are meant to be updated during pre-training with set_wb. Also has methods to
        transform an input_tensor to an encoded representation via the weights/biases
        of the layer.

        :param input_dim: An int representing the dimension of input to this layer.
        :param output_dim: An int representing the dimension of the encoded output.
        :param activation: A function to transform the inputs to this layer (sigmoid, etc.).
        :param weights: A tensor with shape [input_dim, output_dim]
        :param biases: A tensor with shape [output_dim]
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.weights = weights
        self.biases = biases

    def set_wb(self, weights, biases):
        #  FIXME: assert that first dimension of W is input_dim, 2nd is output_dim
        self.weights = weights
        self.biases = biases

    @property
    def is_pretrained(self):
        return not (self.weights is None and self.biases is None)

    def encode(self, input_tensor):
        assert self.is_pretrained, "Cannot encode when not pretrained."
        return self.activation(tf.matmul(input_tensor, self.weights) + self.biases)


class SDAutoencoder:
    """A stacked denoising autoencoder."""

    def check_assertions(self):
        assert 0 <= self.noise <= 1, "Invalid noise value given: %s" % self.noise

    def __init__(self, dims, activations, noise=0.0, loss="cross-entropy",
                 lr=0.0001, batch_size=100, print_step=50):
        """Initializes a Stacked Denoising Autoencoder based on the dimension of each
        layer in the neural network and the activation function of each layer. SDA only
        undergoes parameter setup at initialization. Main functions to utilize the SDA are:

        pretrain_network: (unsupervised) Greedily pre-trains every layer of the neural network,
            beginning with feeding the raw data input to the first layer, and getting an encoded
            version from the output of the first layer. Adjusts parameters of the network (weights and
            biases of each layer) during training, via a stochastic Adam optimization method.

        finetune_parameters: (supervised) Adds a layer of fine-tuning to the network, adjusting
            the weights and biases of all layers simultaneously via a softmax classifier with test
            y-values. Also prints batch accuracy during each print step.

        write_encoded_input: Reads the x-values from a test data source and transforms them
            accordingly through the network (which has all parameters optimized from pre-training).
            Writes the newly represented features to a specified file.

        (Example usage)
            sda = SDAutoencoder([784, 400, 200, 10], ["relu", "relu", "relu"], noise=0.05)
            sda.pretrain_network(X_TRAIN_PATH)
            sda.finetune_parameters(X_TRAIN_PATH, Y_TRAIN_PATH)
            sda.write_encoded_input(your_filename, X_TEST_PATH, input_dim)


        :param dims: A list of ints containing the dimensions of the x-values at each step of
            the network. The first entry is the overall input_dim, and the last entry is the
            overall output_dim from the network.
        :param activations: A list of activation functions for each layer in the network.
        :param noise: A double from 0 to 1 representing the amount of masking on the input (noise).
        :param loss: A string representing the loss function used.
        :param lr: A double representing the learning rate of the optimization method.
        :param batch_size: The number of cases fed to the network in each batch from file.
        :param print_step: The number of batches processed before each print progress step.
        """
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.hidden_layers = self.create_new_layers(dims, activations)

        self.noise = noise
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size
        self.print_step = print_step

        self.check_assertions()
        print("Initialized SDA network with dims %s, noise %s, loss %s, learning rate %s, and batch_size %s."
              % (dims, self.noise, self.loss, self.lr, self.batch_size))

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

    def write_data(self, data, filename):
        """Writes data in data_tensor and appends to the end of filename in csv format

        :param data: A 2-dimensional numpy array
        :param filename: A string representing the save filepath
        :return: None
        """
        with open(filename, "ab") as file:
            np.savetxt(file, data, delimiter=",")

    def write_encoded_input(self, filename, x_test_path, input_dim):
        """Get encoded feature representation.

        :param filename:
        :param x_test_path:
        :param input_dim:
        :return:
        """
        sess = tf.Session()
        x_test = get_next_batch(x_test_path, self.batch_size)
        x_input = tf.placeholder(tf.float32, shape=[None, input_dim])
        x_encoded = self.get_encoded_input(x_input, len(self.hidden_layers))

        for x_batch in x_test:
            self.write_data(sess.run(x_encoded, feed_dict={x_input: x_batch}), filename)

    # def get_encoded_input(self, input_tensor, depth):
    #     """Recursive implementation.
    #
    #     :param input_tensor:
    #     :param depth:
    #     :return:
    #     """
    #     def encoder_helper(current_input_tensor, current_depth, hidden_layer_index):
    #         if current_depth == 0:
    #             return current_input_tensor
    #         encoded = self.hidden_layers[hidden_layer_index].encode(current_input_tensor)
    #         return encoder_helper(encoded, current_depth - 1, hidden_layer_index + 1)
    #     return encoder_helper(input_tensor, depth, 0)

    def get_encoded_input(self, input_tensor, depth):
        for i in range(depth):
            input_tensor = self.hidden_layers[i].encode(input_tensor)
        return input_tensor

    def pretrain_layer(self, depth, batch_generator, act=tf.nn.sigmoid):
        sess = tf.Session()

        print("Starting to pretrain layer %d." % depth)

        hidden_layer = self.hidden_layers[depth]
        input_dim, output_dim = hidden_layer.input_dim, hidden_layer.output_dim

        x_original = tf.placeholder(tf.float32, shape=[None, input_dim])
        x_latent = self.get_encoded_input(x_original, depth)
        x_corrupt = self.corrupt(x_latent)

        encode = {"weights": tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1, dtype=tf.float32)),
                  "biases": tf.Variable(tf.truncated_normal([output_dim], stddev=0.1, dtype=tf.float32))}

        decode = {"weights": tf.transpose(encode["weights"]),  # Tied weights
                  "biases": tf.Variable(tf.truncated_normal([input_dim], stddev=0.1, dtype=tf.float32))}

        encoded = act(tf.matmul(x_corrupt, encode["weights"]) + encode["biases"])
        decoded = tf.matmul(encoded, decode["weights"]) + decode["biases"]

        # Reconstruction loss
        loss = self.get_loss(x_latent, decoded)

        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        sess.run(tf.initialize_all_variables())

        step = 0
        for batch_x_original in batch_generator:  # FIXME: Might need to train much more than one run-through
            sess.run(train_op, feed_dict={x_original: batch_x_original})
            if step % self.print_step == 0:
                loss_value = sess.run(loss, feed_dict={x_original: batch_x_original})
                print("Step %s, batch loss = %s" % (step, loss_value))
            step += 1

        # Set the weights and biases of pretrained hidden layer
        hidden_layer.set_wb(weights=sess.run(encode["weights"]), biases=sess.run(encode["biases"]))  # Need a feed_dict?

        print("Finished pretraining of layer %d. Updated layer weights and biases." % depth)

    def get_loss(self, tensor_1, tensor_2):
        if self.loss == "rmse":
            return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(tensor_1, tensor_2))))
        elif self.loss == "cross-entropy":
            return tf.reduce_mean(-tf.reduce_sum(
                tensor_1 * tf.log(tensor_2) + (1 - tensor_1) * tf.log(1 - tensor_2), reduction_indices=[1]
            ))  # FIXME: Check to verify correctness of math

    def create_new_layers(self, dims, activations):
        """

        :param dims: Ex. [784, 200, 10]
        :param activations: Ex. ['relu', 'relu']
        :return: [NNLayer(input_dim=784, output_dim=200), NNLayer(input_dim=200, output_dim=10)]
        """
        assert set(activations + ALLOWED_ACTIVATIONS) == set(ALLOWED_ACTIVATIONS), "Incorrect activation(s) given."
        assert len(dims) == len(activations) + 1, "Incorrect number of layers/activations."
        return [NNLayer(dims[i], dims[i + 1], activations[i]) for i in range(len(activations))]

    @stopwatch
    def pretrain_network(self, x_train_path):
        for i in range(len(self.hidden_layers)):
            x_train = get_next_batch(x_train_path, self.batch_size)
            self.pretrain_layer(i, x_train, act=tf.nn.sigmoid)

    @stopwatch
    def finetune_parameters(self, x_train_path, y_train_path):
        sess = tf.Session()

        print("Starting to fine tune parameters of network.")

        x = tf.placeholder(tf.float32, self.input_dim)
        x_encoded = self.get_encoded_input(x, depth=len(self.hidden_layers)) # Full depth encoding
        W = tf.Variable(tf.truncated_normal(shape=[500, 2], stddev=0.1))  # FIXME: Make this a parameter
        b = tf.Variable(tf.constant(0.1, shape=[2]))
        y_pred = tf.nn.softmax(tf.matmul(x_encoded, W) + b)

        y_actual = tf.placeholder(tf.float32, [2])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_pred), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cross_entropy)

        x_train = get_next_batch(x_train_path, self.batch_size)
        y_train = get_next_batch(y_train_path, self.batch_size)

        for i in range(1000):  # FIXME: Make a parameter
            batch_xs, batch_ys = next(x_train), next(y_train)
            sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})

            if i % 100 == 0:
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("Step %s, batch accuracy: " % i, sess.run(accuracy, feed_dict={batch_xs, batch_ys}))


def main():
    sda = SDAutoencoder(dims=[3997, 500, 500, 500],
                        activations=["sigmoid", "sigmoid", "sigmoid"],
                        noise=0.05,
                        loss="cross-entropy")

    sda.pretrain_network(X_TRAIN_PATH)
    sda.finetune_parameters(X_TRAIN_PATH, Y_TRAIN_PATH)
    sda.write_encoded_input("../data/transformed.csv", X_TEST_PATH, 3997)


if __name__ == "__main__":
    main()
