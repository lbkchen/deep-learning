from final_sda import get_batch_generator, stopwatch
import tensorflow as tf
import numpy as np


X_TRAIN_PATH = "../data/x_train_transformed_SAM_2.csv"
Y_TRAIN_PATH = "../data/splits/OPYTrainSAM.csv"
X_TEST_PATH = "../data/x_test_transformed_SAM_2.csv"
Y_TEST_PATH = "../data/splits/OPYTestSAM.csv"
OUTPUT_PATH = "../data/ip_predicted_ys_1_epoch.csv"


def average(lst):
    return sum(lst) / len(lst)


def append_with_limit(lst, val, limit=10):
    """Non-destructive function that returns a copy of the original list with the appended value and limit."""
    lst_copy = lst[:]
    lst_copy.append(val)
    return lst_copy[-limit:]


def write_data(data, filename):  # FIXME: Copied from sda, should refactor to static
    """Writes data in data_tensor and appends to the end of filename in csv format.

    :param data: A 2-dimensional numpy array.
    :param filename: A string representing the save filepath.
    :return: None
    """
    with open(filename, "ab") as file:
        np.savetxt(file, data, delimiter=",")


@stopwatch
def train_softmax(input_dim, output_dim, x_train_filepath, y_train_filepath, lr=0.0001, batch_size=100,
                  print_step=50, epochs=1):
    """Trains a softmax model for prediction."""
    # Model input and parameters
    x = tf.placeholder(tf.float32, [None, input_dim])
    weights = tf.Variable(tf.truncated_normal(shape=[input_dim, output_dim], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[output_dim]))

    # Outputs and true y-values
    y_logits = tf.matmul(x, weights) + biases
    y_pred = tf.nn.softmax(y_logits)
    y_actual = tf.placeholder(tf.float32, [None, output_dim])

    # Cross entropy and training step
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_actual))
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    # Start session and run batches based on number of epochs
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    x_train = get_batch_generator(filename=x_train_filepath, batch_size=batch_size,
                                  skip_header=False, repeat=epochs - 1)
    y_train = get_batch_generator(filename=y_train_filepath, batch_size=batch_size,
                                  skip_header=True, repeat=epochs - 1)
    step = 0
    accuracy_history = []
    for batch_xs, batch_ys in zip(x_train, y_train):
        sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})

        # Debug
        # if step == 100:
        #     break

        # Assess training accuracy for current batch
        if step % print_step == 0:
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_val = sess.run(accuracy, feed_dict={x: batch_xs, y_actual: batch_ys})
            print("Step %s, current batch training accuracy: %s" % (step, accuracy_val))
            accuracy_history = append_with_limit(accuracy_history, accuracy_val)

        # Assess training accuracy for last 10 batches
        if step > 0 and step % (print_step * 10) == 0:
            print("Predicted y-values:\n", sess.run(y_pred, feed_dict={x: batch_xs}))
            print("Overall batch training accuracy for steps %s to %s: %s" % (step - 10 * print_step,
                                                                              step,
                                                                              average(accuracy_history)))

        step += 1

    parameters_dict = {
        "weights": sess.run(weights),
        "biases": sess.run(biases)
    }
    sess.close()
    return parameters_dict


@stopwatch
def test_model(parameters_dict, input_dim, output_dim, x_test_filepath, y_test_filepath, output_filepath,
               batch_size=100, print_step=100):
    """

    :param parameters_dict: Must contain keys 'weights' and 'biases' with their respective values
    :param input_dim:
    :param output_dim:
    :param x_test_filepath:
    :param y_test_filepath:
    :param output_filepath:
    :param batch_size:
    :param print_step:
    :return:
    """
    # Model input and parameters
    x = tf.placeholder(tf.float32, [None, input_dim])
    weights = parameters_dict["weights"]
    biases = parameters_dict["biases"]

    # Outputs and true y-values
    y_pred = tf.nn.softmax(tf.matmul(x, weights) + biases)
    y_actual = tf.placeholder(tf.float32, [None, output_dim])

    # Evaluate testing accuracy
    sess = tf.Session()
    x_test = get_batch_generator(filename=x_test_filepath, batch_size=batch_size, skip_header=False)
    y_test = get_batch_generator(filename=y_test_filepath, batch_size=batch_size, skip_header=True)
    step = 0
    accuracy_history = []
    for batch_xs, batch_ys in zip(x_test, y_test):
        write_data(data=sess.run(y_pred, feed_dict={x: batch_xs}), filename=output_filepath)

        # Debug
        # if step == 100:
        #     break

        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_val = sess.run(accuracy, feed_dict={x: batch_xs, y_actual: batch_ys})
        accuracy_history.append(accuracy_val)

        if step % print_step == 0:
            print("Step %s, current batch testing accuracy: %s" % (step, accuracy_val))

        if step > 0 and step % (print_step * 10) == 0:
            print("Predicted y-values:\n", sess.run(y_pred, feed_dict={x: batch_xs}))

        step += 1
    sess.close()
    print("Testing complete and written to %s, overall accuracy: %s" % (output_filepath, average(accuracy_history)))


@stopwatch
def main():
    trained_parameters = train_softmax(input_dim=500,
                                       output_dim=2,
                                       x_train_filepath=X_TRAIN_PATH,
                                       y_train_filepath=Y_TRAIN_PATH,
                                       epochs=1)

    test_model(parameters_dict=trained_parameters,
               input_dim=500,
               output_dim=2,
               x_test_filepath=X_TEST_PATH,
               y_test_filepath=Y_TEST_PATH,
               output_filepath=OUTPUT_PATH)


if __name__ == "__main__":
    main()
