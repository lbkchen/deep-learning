import tf.final_sda as sda
import tensorflow as tf
import numpy as np


def append_with_limit(lst, val, limit=10):
    """Non-destructive function that returns a copy of the original list with the appended value and limit."""
    lst_copy = lst[:]
    lst_copy.append(val)
    return lst_copy[-limit:]


@sda.stopwatch
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
    x_train = sda.get_batch_generator(filename=x_train_filepath, batch_size=batch_size, repeat=epochs - 1)
    y_train = sda.get_batch_generator(filename=y_train_filepath, batch_size=batch_size, repeat=epochs - 1)
    step = 0
    accuracy_history = []
    for batch_xs, batch_ys in zip(x_train, y_train):
        sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})

        # Assess training accuracy for current batch
        if step % print_step == 0:
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_val = sess.run(accuracy, feed_dict={x: batch_xs, y_actual: batch_ys})
            print("Step %s, current batch accuracy: %s" % (step, accuracy_val))
            accuracy_history = append_with_limit(accuracy_history, accuracy_val)

        # Assess training accuracy for last 10 batches
        if step > 0 and step % (print_step * 10) == 0:
            print("Predicted y-values:\n", sess.run(y_pred, feed_dict={x: batch_xs}))
            print("Overall batch accuracy for steps %s to %s" % (step - 10 * batch_size, step))

        step += 1

    parameters_dict = {
        "weights": sess.run(weights),
        "biases": sess.run(biases)
    }

    return parameters_dict


@sda.stopwatch
def test_model(x_test_filepath, y_test_filepath):
    pass


@sda.stopwatch
def main():
    pass


if __name__ == "__main__":
    main()