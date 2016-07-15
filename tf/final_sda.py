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


allowed_activations = ["sigmoid", "tanh", "relu", "softmax"]
allowed_losses = ["rmse", "cross-entropy"]

X_TRAIN_PATH = "../data/splits/PXTrainSAM.csv"
Y_TRAIN_PATH = "../data/splits/PYTrainSAM.csv"
X_TEST_PATH = "../data/splits/PXTestSAM.csv"
Y_TEST_PATH = "../data/splits/YTestSAM.csv"

# xs_filepath = "../data/S01X.csv"
# ys_filepath = "../data/S01Y.csv"


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
    :param batch_size:
    :return:
    """
    with open(filename, "rt") as file:
        reader = csv.reader(file)
        index = 0
        this_batch = []
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

    def __init__(self, input_dim, output_dim, activation=None, weights=None, biases=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.weights = weights
        self.biases = biases

    def set_wb(self, weights, biases):
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

    # def check_assertions(self):
    #     global allowed_activations, allowed_losses
    #     assert self.loss in allowed_losses, "Incorrect loss given."
    #     assert 'list' in str(type(self.dims)), "dims must be a list even if there is one layer."
    #     assert len(self.epochs) == len(self.dims), "No. of epochs must equal to no. of hidden layers."
    #     assert len(self.activations) == len(self.dims), "No. of activations must equal to no. of hidden layers."
    #     assert all(True if x > 0 else False for x in self.epochs), "No. of epoch must be at least 1."
    #     assert set(self.activations + allowed_activations) == set(allowed_activations), "Incorrect activation given."
    #     assert 0 <= self.noise <= 1, "Invalid noise value given: %s" % self.noise

    def check_assertions(self):
        global allowed_activations, allowed_losses
        assert 0 <= self.noise <= 1, "Invalid noise value given: %s" % self.noise

    def __init__(self, dims, activations, noise=0.0, loss="cross-entropy",
                 lr=0.0001, batch_size=100, print_step=50):
        """
        Example: sda = SDAutoencoder([784, 400, 200, 100], ["relu", "relu", "relu"], [200, 200, 200])

        :param dims:
        :param activations:
        :param epochs:
        :param noise:
        :param loss:
        :param lr:
        :param batch_size:
        :param print_step:
        """
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.hidden_layers = self.create_new_layers(dims, activations)
        # self.dims = dims
        # self.activations = activations
        self.noise = noise
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size
        self.print_step = print_step

        self.check_assertions()
        # self.weights = []
        # self.biases = []

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

        hidden_layer = self.hidden_layers[depth]
        input_dim, output_dim = hidden_layer.input_dim, hidden_layer.output_dim

        x_original = tf.placeholder(tf.float32, shape=[None, input_dim])
        x_latent = self.get_encoded_input(x_original, depth)
        x_corrupt = self.corrupt(x_latent)

        encode = {"weights": tf.Variable(tf.truncated_normal([input_dim, output_dim], dtype=tf.float32)),
                  "biases": tf.Variable(tf.truncated_normal([output_dim], dtype=tf.float32))}

        decode = {"weights": tf.transpose(encode["weights"]),
                  "biases": tf.Variable(tf.truncated_normal([input_dim], dtype=tf.float32))}

        encoded = act(tf.matmul(x_corrupt, encode["weights"]) + encode["biases"])
        decoded = tf.matmul(encoded, decode["weights"]) + decode["biases"]

        # Reconstruction loss
        loss = self.get_loss(x_latent, decoded)

        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        sess.run(tf.initialize_all_variables())

        step = 0
        for batch_x_original in batch_generator:
            sess.run(train_op, feed_dict={x_original: batch_x_original})
            if (step + 1) % self.print_step == 0:
                loss_value = sess.run(loss, feed_dict={x_original: batch_x_original})
                print("Batch loss = %s" % loss_value)
            step += 1

        # Set the weights and biases of pretrained hidden layer
        hidden_layer.weights = sess.run(encode["weights"])
        hidden_layer.biases = sess.run(encode["biases"])

    def get_loss(self, tensor_1, tensor_2):
        if self.loss == "rmse":
            return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(tensor_1, tensor_2))))
        elif self.loss == "cross-entropy":
            return tf.reduce_mean(-tf.reduce_sum(
                tensor_1 * tf.log(tensor_2) + (1 - tensor_1) * tf.log(1 - tensor_2), reduction_indices=[1]
            ))

    def create_new_layers(self, dims, activations):
        assert set(activations + allowed_activations) == set(allowed_activations), "Incorrect activation(s) given."
        assert len(dims) == len(activations) + 1, "Incorrect number of layers/activations."
        return [NNLayer(dims[i], dims[i + 1], activations[i]) for i in range(len(activations))]

    def pretrain_network(self, x_train_path, y_train_path, x_test_path):
        for i in range(len(self.hidden_layers)):
            x_train = get_next_batch(x_train_path, self.batch_size)
            y_train = get_next_batch(y_train_path, self.batch_size)
            x_test = get_next_batch(x_test_path, self.batch_size)
            self.pretrain_layer(i, x_train, act=tf.nn.sigmoid)

    def finetune_parameters(self, x_train_path, y_train_path):
        sess = tf.Session()
        x = tf.placeholder(tf.float32, self.input_dim)
        x_encoded = self.get_encoded_input(x, depth=len(self.hidden_layers)) # Full depth encoding
        W = tf.Variable(tf.truncated_normal(shape=[500, 2], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[2]))
        y_pred = tf.nn.softmax(tf.matmul(x_encoded, W) + b)

        y_actual = tf.placeholder(tf.float32, [2])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_pred), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cross_entropy)

        x_train = get_next_batch(x_train_path, self.batch_size)
        y_train = get_next_batch(y_train_path, self.batch_size)

        for i in range(1000):
            batch_xs, batch_ys = next(x_train), next(y_train)
            sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})

            if i % 100 == 0:
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("Batch accuracy: ", sess.run(accuracy, feed_dict={batch_xs, batch_ys}))


def main():
    sda = SDAutoencoder(dims=[3997, 500, 500, 500],
                        activations=["sigmoid", "sigmoid", "sigmoid"],
                        noise=0.05,
                        loss="cross-entropy")

    sda.pretrain_network(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH)
    sda.finetune_parameters(X_TRAIN_PATH, Y_TRAIN_PATH)


if __name__ == "__main__":
    main()
