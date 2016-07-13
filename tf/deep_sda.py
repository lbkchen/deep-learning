"""
Stacked Denoising Autoencoder implementation

Ken Chen
"""

import tensorflow as tf
import numpy as np
import math
import time
import csv
from functools import wraps
from sklearn.preprocessing import MaxAbsScaler


"""
###########################
### SETUP AND VARIABLES ###
###########################
"""


allowed_activations = ["sigmoid", "tanh", "relu", "softmax"]
allowed_noises = [None, "gaussian", "mask"]
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
        print("Total time elapsed for execution of %s:" %f, elapsed_time)
        return result

    return wrapped


"""
##################################
### HELPER CLASSES / FUNCTIONS ###
##################################
"""


def get_batch(X, X_, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X_[a]

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
        """Creates the autoencoder with some parameters

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
            frac = float(self.noise.split("-")[1]) # Self.noise can have form ex. mask-0.4
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

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def run(self, data_x, data_x_, hidden_dim, activation, loss,
            lr, print_step, epoch, batch_size=100):
        input_dim = len(data_x[0])
        sess = tf.Session()

        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name="x")
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name="x_")

        encode = {"weights": tf.Variable(tf.truncated_normal([input_dim, hidden_dim], dtype=tf.float32)),
                  "biases": tf.Variable(tf.truncated_normal([hidden_dim], dtype=tf.float32))}

        decode = {"weights": tf.transpose(encode["weights"]),
                  "biases": tf.Variable(tf.truncated_normal([input_dim], dtype=tf.float32))}

        encoded = self.activate(tf.matmul(x, encode["weights"]) + encode["biases"], activation)
        decoded = tf.matmul(encoded, decode["weights"]) + decode["biases"]

        # Reconstruction loss
        if loss == "rmse":
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x_, decoded))))
        elif loss == "cross-entropy":
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(decoded, x_))
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        sess.run(tf.initialize_all_variables())

        for i in range(epoch):
            b_x, b_x_ = None, None #FIXME add batch distribution system of data
            sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            if (i + 1) % print_step == 0:
                loss_value = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
                print("Epoch %d: global loss = %s" % (i, loss_value))

        # debug
        # print('Decoded', sess.run(decoded, feed_dict={x: self.data_x_})[0])
        self.weights.append(sess.run(encode["weights"]))
        self.biases.append(sess.run(encode["biases"]))
        return sess.run(encoded, feed_dict={x: data_x_})

    def activate(self, linear, name):
        if name == "sigmoid":
            return tf.nn.sigmoid(linear, name="encoded")
        elif name == "tanh":
            return tf.nn.tanh(linear, name="encoded")
        elif name == "relu":
            return tf.nn.relu(linear, name="encoded")
        elif name == "softmax":
            return tf.nn.softmax(linear, name="encoded")
        else:
            return linear

def main():
    # xs = np.genfromtxt("../data/S01X.csv", delimiter=",")
    # ys = np.genfromtxt("../data/S01Y.csv", delimiter=",")
    # half = len(xs) // 2
    #
    # train_x = MaxAbsScaler().fit_transform(xs[:half, :])
    # train_y = ys[:half , :]
    # test_x = MaxAbsScaler().fit_transform(xs[half:, :])
    # test_y = ys[half:, :]
    #
    # model = SDAutoencoder(dims=[4000, 2000],
    #                       activations=["sigmoid", "sigmoid"],
    #                       epoch=[300, 300],
    #                       loss="rmse",
    #                       lr=0.001,
    #                       batch_size=1500,
    #                       print_step=50)
    #
    # xx = model.fit_transform(np.r_[train_x, test_x])
    a = get_next_batch(ys_filepath, 25)
    print(next(a))
    print(next(a))

if __name__ == "__main__":
    main()
