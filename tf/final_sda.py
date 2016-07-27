"""
Stacked Denoising Autoencoder Implementation

Ken Chen
"""

import tensorflow as tf
import numpy as np
import time
import csv
from functools import wraps


"""
###########################
### SETUP AND CONSTANTS ###
###########################
"""


ALLOWED_ACTIVATIONS = ["sigmoid", "tanh", "relu", "softmax"]
ALLOWED_LOSSES = ["rmse", "cross-entropy"]

# X_TRAIN_PATH = "../data/splits/PXTrainSAM.csv"
# Y_TRAIN_PATH = "../data/splits/OPYTrainSAM.csv"
# X_TEST_PATH = "../data/splits/PXTestSAM.csv"
# Y_TEST_PATH = "../data/splits/OPYTestSAM.csv"

# X_TRAIN_PATH = "../data/splits/small/PXTrainSAMsmall.csv"
# Y_TRAIN_PATH = "../data/splits/small/OPYTrainSAMsmall.csv"
# X_TEST_PATH = "../data/splits/small/PXTestSAMsmall.csv"
# Y_TEST_PATH = "../data/splits/small/YTestSAMsmall.csv"
# ENCODED_X_PATH = "../data/x_test_transformed_SAM.csv"

X_TRAIN_PATH = "../data/rose/SAMPart01_train_x_r.csv"
Y_TRAIN_PATH = "../data/rose/SAMPart01_train_y_r.csv"
X_TEST_PATH = "../data/rose/SAMPart01_test_x_r.csv"
Y_TEST_PATH = "../data/rose/SAMPart01_test_y_r.csv"
ENCODED_X_PATH = "../data/x_test_new_rose.csv"

TENSORBOARD_LOGDIR = "../logs/tensorboard"
TENSORBOARD_LOG_STEP = 10


"""
##################
### DECORATORS ###
##################
"""


def stopwatch(f):
    """Simple decorator that prints the execution time of a function."""

    @wraps(f)
    def wrapped(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print("Total seconds elapsed for execution of %s:" % f, elapsed_time)
        return result

    return wrapped


"""
##################################
### HELPER CLASSES / FUNCTIONS ###
##################################
"""


def get_batch_generator(filename, batch_size, skip_header=True, repeat=0):
    """Generator that gets the net batch of batch_size x or y values
    from the given file.

    :param filename: A string of the file to write to.
    :param batch_size: An int: the number of lines to include in each batch.
    :param skip_header: If True, then skips the first line of the file.
    :param repeat: An int specifying the number of times to repeat going through
        the file. Repeat of 2 will return a generator that iterates through the
        full file three times before stopping iteration.
    :return:
    """
    assert repeat < 1000, "Recursion depth will be exceeded."
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

        # Catch any remainders in current data set
        if this_batch:
            yield this_batch

        print("Finished a batch iteration through %s" % filename)
        if repeat > 0:
            for item in get_batch_generator(filename, batch_size, skip_header, repeat - 1):
                yield item


def attach_variable_summaries(var, name, summ_list):
    """Attach statistical summaries to a tensor for tensorboard visualization."""
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        summ_mean = tf.scalar_summary("mean/" + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(var, mean))))
        summ_std = tf.scalar_summary('stddev/' + name, stddev)
        summ_max = tf.scalar_summary('max/' + name, tf.reduce_max(var))
        summ_min = tf.scalar_summary('min/' + name, tf.reduce_min(var))
        summ_hist = tf.histogram_summary(name, var)
    summ_list.extend([summ_mean, summ_std, summ_max, summ_min, summ_hist])


def attach_scalar_summary(var, name, summ_list):
    """Attach scalar summaries to a scalar."""
    summ = tf.scalar_summary(tags=name, values=var)
    summ_list.append(summ)


class NNLayer:
    """A container class to represent a hidden layer in the autoencoder network."""

    def __init__(self, input_dim, output_dim, name="hidden_layer", activation=None, weights=None, biases=None):
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
        self.name = name
        self.activation = activation
        self.weights = weights
        self.biases = biases
        self._weights = None  # Weights Variable
        self._biases = None  # Biases Variable

    def set_wb(self, weights, biases):
        """Used during pre-training for convenience."""
        self.weights = weights
        self.biases = biases

        print("Set weights of layer with shape", tf.shape(weights))
        print("Set biases of layer with shape", tf.shape(weights))

    def set_wb_variables(self, summ_list):
        """This function is called at the beginning of supervised fine tuning to create new
        variables with initial values based on their static parameter counterparts. These
        variables can then all be adjusted simultaneously during the fine tune optimization."""
        assert self.is_pretrained, "Cannot set Variables when not pretrained."
        with tf.name_scope(self.name):
            self._weights = tf.Variable(self.weights, dtype=tf.float32, name="weights")
            self._biases = tf.Variable(self.biases, dtype=tf.float32, name="biases")
            attach_variable_summaries(self._weights, name=self._weights.name, summ_list=summ_list)
            attach_variable_summaries(self._biases, name=self._biases.name, summ_list=summ_list)
        print("Created new weights and bias variables from current values.")

    def update_wb(self, sess):
        """This function is called at the end of supervised fine tuning to update the static
        weight and bias values to the newest snapshot of their dynamic variable counterparts."""
        assert self._weights is not None and self._biases is not None, "Weights and biases Variables not set."
        self.weights = sess.run(self._weights)
        self.biases = sess.run(self._biases)
        print("Updated weights and biases with corresponding evaluated variable values.")

    def get_weight_variable(self):
        return self._weights

    def get_bias_variable(self):
        return self._biases

    @property
    def is_pretrained(self):
        return self.weights is not None and self.biases is not None

    def encode(self, input_tensor, use_variables=False):
        """Performs this layer's encoding on the input_tensor. use_variables is set to true
        during the fine-tuning stage, when all parameters of each layer need to be adjusted."""
        assert self.is_pretrained, "Cannot encode when not pre-trained."
        if use_variables:
            return self.activate(tf.matmul(input_tensor, self._weights) + self._biases)
        else:
            return self.activate(tf.matmul(input_tensor, self.weights) + self.biases)

    def activate(self, input_tensor, name=None):
        """Applies the activation function for this layer based on self.activation."""
        if self.activation == "sigmoid":
            return tf.nn.sigmoid(input_tensor, name=name)
        if self.activation == "tanh":
            return tf.nn.tanh(input_tensor, name=name)
        if self.activation == "relu":
            return tf.nn.relu(input_tensor, name=name)
        if self.activation == "softmax":
            return tf.nn.softmax(input_tensor, name=name)
        else:
            print("Activation function not valid. Using the identity.")
            return input_tensor


"""
#####################################
### STACKED DENOISING AUTOENCODER ###
#####################################
"""


class SDAutoencoder:
    """A stacked denoising autoencoder."""

    def check_assertions(self):
        assert 0 <= self.noise <= 1, "Invalid noise value given: %s" % self.noise
        assert self.loss in ALLOWED_LOSSES

    def __init__(self, dims, activations, sess, noise=0.0, loss="cross-entropy",
                 lr=0.001, batch_size=100, print_step=100):
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
            sda.write_encoded_input(your_filename, X_TEST_PATH)

        :param dims: A list of ints containing the dimensions of the x-values at each step of
            the network. The first entry is the overall input_dim, and the last entry is the
            overall output_dim from the network.
        :param activations: A list of activation functions for each layer in the network.
        :param sess: A tf.Session to be used by the autoencoder
        :param noise: A double from 0 to 1 representing the amount of masking on the input (noise).
        :param loss: A string representing the loss function used.
        :param lr: A double representing the learning rate of the optimization method.
        :param batch_size: The number of cases fed to the network in each batch from file.
        :param print_step: The number of batches processed before each print progress step.
        """
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.hidden_layers = self.create_new_layers(dims, activations)
        self.sess = sess

        self.noise = noise
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size
        self.print_step = print_step

        self.check_assertions()
        print("Initialized SDA network with dims %s, noise %s, loss %s, learning rate %s, and batch_size %s."
              % (dims, self.noise, self.loss, self.lr, self.batch_size))

    @property
    def is_pretrained(self):
        """Returns whether the whole autoencoder network (all layers) is pre-trained."""
        return all([layer.is_pretrained for layer in self.hidden_layers])

    def get_all_variables(self, additional_vars=None):
        """Returns all trainable variables of the neural network."""
        all_vars = []
        for layer in self.hidden_layers:
            all_vars.extend([layer.get_weight_variable(), layer.get_bias_variable()])
        if additional_vars:
            all_vars.extend(additional_vars)
        return all_vars

    def setup_all_variables(self, summ_list):
        """See NNLayer.set_wb_variables. Performs layer method on all hidden layers."""
        for layer in self.hidden_layers:
            layer.set_wb_variables(summ_list)

    def finalize_all_variables(self):
        """See NNLayer.finalize_all_variables. Performs layer method on all hidden layers."""
        for layer in self.hidden_layers:
            layer.update_wb(self.sess)

    def corrupt(self, tensor, corruption_level=0.05):
        """Uses the masking noise algorithm to mask corruption_level proportion
        of the input.

        :param tensor: A tensor whose values are to be corrupted.
        :param corruption_level: An int [0, 1] specifying the probability to corrupt each value.
        :return: The corrupted tensor.
        """
        total_samples = tf.reduce_prod(tf.shape(tensor))
        corruption_matrix = tf.multinomial(tf.log([[corruption_level, 1 - corruption_level]]), total_samples)
        corruption_matrix = tf.cast(tf.reshape(corruption_matrix, shape=tf.shape(tensor)), dtype=tf.float32)
        return tf.mul(tensor, corruption_matrix)

    def save_variables(self, filepath):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, filepath)
        print("Model saved in file: %s" % save_path)

    def write_data(self, data, filename):  # FIXME: Should be a static function and outside of class
        """Writes data in data_tensor and appends to the end of filename in csv format.

        :param data: A 2-dimensional numpy array.
        :param filename: A string representing the save filepath.
        :return: None
        """
        with open(filename, "ab") as file:
            np.savetxt(file, data, delimiter=",")

    @stopwatch
    def write_encoded_input(self, filepath, x_test_path):
        """Get encoded feature representation and writes to filepath.

        :param filepath: A string specifying the file path/name to write the encoded input to.
        :param x_test_path: A string specifying the file path of the x test values.
        :return: None
        """
        sess = self.sess
        x_test = get_batch_generator(x_test_path, self.batch_size, skip_header=True)
        x_input = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        x_encoded = self.get_encoded_input(x_input, len(self.hidden_layers), use_variables=False)

        print("Beginning to write to file.")
        for x_batch in x_test:
            self.write_data(sess.run(x_encoded, feed_dict={x_input: x_batch}), filepath)
        print("Written encoded input to file %s" % filepath)

    def get_encoded_input(self, input_tensor, depth, use_variables=False):
        """Performs an encoding on input_tensor through the neural network depending on depth.
        If depth is 0, then input_tensor is simply returned. If depth is 3, then input_tensor
        will be encoded through the first three layers of the network. If depth is -1, then
        input_tensor will be encoded through the entire network.

        :param input_tensor: A tensor to encode.
        :param depth: The number of layers through which input_tensor will be encoded. If -1,
            then the full network encoding will be used.
        :param use_variables: A boolean representing whether to use tf.Variable representations
            of layer parameters. This is set to True only during the fine-tuning stage.
        :return: The encoded input_tensor.
        """
        depth = len(self.hidden_layers) if depth == -1 else depth
        for i in range(depth):
            input_tensor = self.hidden_layers[i].encode(input_tensor, use_variables=use_variables)
        return input_tensor

    def pretrain_layer(self, depth, batch_generator, act=tf.nn.sigmoid):
        sess = self.sess

        print("Starting to pretrain layer %d." % depth)
        hidden_layer = self.hidden_layers[depth]
        summary_list = []

        with tf.name_scope(hidden_layer.name):
            input_dim, output_dim = hidden_layer.input_dim, hidden_layer.output_dim

            with tf.name_scope("x_values"):
                x_original = tf.placeholder(tf.float32, shape=[None, self.input_dim])
                x_latent = self.get_encoded_input(x_original, depth, use_variables=False)
                x_corrupt = self.corrupt(x_latent, corruption_level=self.noise)

            with tf.name_scope("encoding_vars"):
                encode = {
                    "weights": tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1, dtype=tf.float32),
                                           name="weights"),
                    "biases": tf.Variable(tf.truncated_normal([output_dim], stddev=0.1, dtype=tf.float32),
                                          name="biases")
                }
                attach_variable_summaries(encode["weights"], encode["weights"].name, summ_list=summary_list)
                attach_variable_summaries(encode["biases"], encode["biases"].name, summ_list=summary_list)

            with tf.name_scope("decoding_vars"):
                decode = {
                    "weights": tf.transpose(encode["weights"],
                                            name="transposed_weights"),  # Tied weights
                    "biases": tf.Variable(tf.truncated_normal([input_dim], stddev=0.1, dtype=tf.float32),
                                          name="decode_biases")
                }
                attach_variable_summaries(decode["weights"], decode["weights"].name, summ_list=summary_list)
                attach_variable_summaries(decode["biases"], decode["biases"].name, summ_list=summary_list)

            with tf.name_scope("encoded_and_decoded"):
                encoded = act(tf.matmul(x_corrupt, encode["weights"]) + encode["biases"])  # FIXME: Need some histogram summaries?
                decoded = tf.matmul(encoded, decode["weights"]) + decode["biases"]
                attach_variable_summaries(encoded, "encoded", summ_list=summary_list)
                attach_variable_summaries(decoded, "decoded", summ_list=summary_list)

            # Reconstruction loss
            with tf.name_scope("reconstruction_loss"):
                loss = self.get_loss(x_latent, decoded)
                attach_scalar_summary(loss, "%s_loss" % self.loss, summ_list=summary_list)

            trainable_vars = [encode["weights"], encode["biases"], decode["biases"]]
            # Only optimize variables for this layer ("greedy")
            with tf.name_scope("train_step"):
                train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss, var_list=trainable_vars)
            sess.run(tf.initialize_all_variables())

            # Merge summaries and get a summary writer
            merged = tf.merge_summary(summary_list)
            pretrain_writer = tf.train.SummaryWriter(TENSORBOARD_LOGDIR + "/train/" + hidden_layer.name, sess.graph)

            step = 0
            for batch_x_original in batch_generator:  # FIXME: Might need to train much more than one run-through
                sess.run(train_op, feed_dict={x_original: batch_x_original})

                # Debug: remove
                # if step >= 50:
                #     break

                if step % self.print_step == 0:
                    loss_value = sess.run(loss, feed_dict={x_original: batch_x_original})
                    print("Step %s, batch %s loss = %s" % (step, self.loss, loss_value))

                if step % TENSORBOARD_LOG_STEP == 0:
                    summary = sess.run(merged, feed_dict={x_original: batch_x_original})
                    pretrain_writer.add_summary(summary, global_step=step)

                step += 1

            # Set the weights and biases of pretrained hidden layer
            hidden_layer.set_wb(weights=sess.run(encode["weights"]), biases=sess.run(encode["biases"]))
            print("Finished pretraining of layer %d. Updated layer weights and biases." % depth)
            # pretrain_writer.flush()
            # pretrain_writer.close()

    def get_loss(self, tensor_1, tensor_2):
        if self.loss == "rmse":
            return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(tensor_1, tensor_2))))
        elif self.loss == "cross-entropy":
            return tf.reduce_mean(-tf.reduce_sum(
                tensor_1 * tf.log(tensor_2) + (1 - tensor_1) * tf.log(1 - tensor_2), reduction_indices=[1]
            ))  # FIXME: Check to verify correctness of math

    def create_new_layers(self, dims, activations):
        """Creates and sets up template layers (un-pretrained) for the network based on dimensions
        and activation functions.

        :param dims: Ex. [784, 200, 10]
        :param activations: Ex. ['relu', 'relu']
        :return: [NNLayer(input_dim=784, output_dim=200), NNLayer(input_dim=200, output_dim=10)]
        """
        assert set(activations + ALLOWED_ACTIVATIONS) == set(ALLOWED_ACTIVATIONS), "Incorrect activation(s) given."
        assert len(dims) == len(activations) + 1, "Incorrect number of layers/activations."
        return [NNLayer(dims[i], dims[i + 1], "hidden_layer_" + str(i), activations[i])
                for i in range(len(activations))]

    def pretrain_network_from_file(self, x_train_path, epochs=1):
        pass

    def pretrain_network_gen(self, x_train_f, epochs=1):
        """

        :param x_train_f: A function that when called with no arguments, returns a generator that
            iterates through the x-train values.
        :param epochs:
        :return:
        """
        print("Starting to pretrain autoencoder network.")
        for i in range(len(self.hidden_layers)):
            x_train = x_train_f()
            self.pretrain_layer(i, x_train, act=tf.nn.sigmoid)
        print("Finished pretraining of autoencoder network.")

    @stopwatch
    def pretrain_network(self, x_train_path, epochs=1):
        print("Starting to pretrain autoencoder network.")
        for i in range(len(self.hidden_layers)):
            x_train = get_batch_generator(x_train_path, self.batch_size, skip_header=True, repeat=epochs-1)
            self.pretrain_layer(i, x_train, act=tf.nn.sigmoid)
        print("Finished pretraining of autoencoder network.")

    @stopwatch
    def finetune_parameters(self, x_train_path, y_train_path, output_dim, epochs=1):
        sess = self.sess
        summary_list = []

        print("Starting to fine tune parameters of network.")
        with tf.name_scope("finetuning"):
            self.setup_all_variables(summary_list)

            with tf.name_scope("inputs"):
                x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="raw_input")
                with tf.name_scope("fully_encoded"):
                    x_encoded = self.get_encoded_input(x, depth=-1, use_variables=True)  # Full depth encoding

            """Note on W below: The difference between self.output_dim and output_dim is that the former
            is the output dimension of the autoencoder stack, which is the dimension of the new feature
            space. The latter is the dimension of the y value space for classification. Ex: If the output
            should be binary, then the output_dim = 2."""
            with tf.name_scope("softmax_variables"):
                W = tf.Variable(tf.truncated_normal(shape=[self.output_dim, output_dim], stddev=0.1), name="weights")
                b = tf.Variable(tf.constant(0.1, shape=[output_dim]), name="biases")
                attach_variable_summaries(W, W.name, summ_list=summary_list)
                attach_variable_summaries(b, b.name, summ_list=summary_list)

            with tf.name_scope("outputs"):
                y_logits = tf.matmul(x_encoded, W) + b
                with tf.name_scope("predicted"):
                    y_pred = tf.nn.softmax(y_logits, name="y_pred")
                    attach_variable_summaries(y_pred, y_pred.name, summ_list=summary_list)
                with tf.name_scope("actual"):
                    y_actual = tf.placeholder(tf.float32, shape=[None, output_dim], name="y_actual")
                    attach_variable_summaries(y_actual, y_actual.name, summ_list=summary_list)

            with tf.name_scope("cross_entropy"):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_logits, y_actual))
                attach_scalar_summary(cross_entropy, "cross_entopy", summ_list=summary_list)

            trainable_vars = self.get_all_variables(additional_vars=[W, b])
            with tf.name_scope("train_step"):
                train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
                    cross_entropy, var_list=trainable_vars)

            with tf.name_scope("evaluation"):
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                attach_scalar_summary(accuracy, "finetune_accuracy", summ_list=summary_list)

            sess.run(tf.initialize_all_variables())

            x_train = get_batch_generator(x_train_path, self.batch_size, skip_header=True, repeat=epochs - 1)
            y_train = get_batch_generator(y_train_path, self.batch_size, skip_header=True, repeat=epochs - 1)

            # Merge summaries and get a summary writer
            merged = tf.merge_summary(summary_list)
            train_writer = tf.train.SummaryWriter(TENSORBOARD_LOGDIR + "/train/finetune", sess.graph)

            step = 0
            for batch_xs, batch_ys in zip(x_train, y_train):
                sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})

                # Debug: remove
                # if step >= 50:
                #     break

                if step % self.print_step == 0:
                    print("Step %s, batch accuracy: " % step,
                          sess.run(accuracy, feed_dict={x: batch_xs, y_actual: batch_ys}))

                if step % (self.print_step * 10) == 0:
                    print("Predicted y-values:", sess.run(y_pred, feed_dict={x: batch_xs}))

                if step % TENSORBOARD_LOG_STEP == 0:
                    summary = sess.run(merged, feed_dict={x: batch_xs, y_actual: batch_ys})
                    train_writer.add_summary(summary, global_step=step)

                step += 1

            self.finalize_all_variables()
            print("Completed fine-tuning of parameters.")
            tuned_params = {"weights": sess.run(W), "biases": sess.run(b)}
            # train_writer.flush()
            # train_writer.close()
            return tuned_params


def main():
    sess = tf.Session()
    sda = SDAutoencoder(dims=[3997, 500, 500, 500],
                        activations=["sigmoid", "sigmoid", "sigmoid"],
                        sess=sess,
                        noise=0.05,
                        loss="rmse",
                        print_step=50)

    sda.pretrain_network(X_TRAIN_PATH)
    sda.finetune_parameters(X_TRAIN_PATH, Y_TRAIN_PATH, output_dim=2)
    sda.write_encoded_input(ENCODED_X_PATH, X_TEST_PATH)


if __name__ == "__main__":
    main()
