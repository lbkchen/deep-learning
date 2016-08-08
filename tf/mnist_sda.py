"""
Example testing SDA model on MNIST digits.
"""

from final_sda import get_batch_generator, SDAutoencoder
from softmax import test_model_gen, test_model
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def get_mnist_batch_generator(is_train, batch_size, batch_limit=100):
    if is_train:
        for _ in range(batch_limit):
            yield mnist.train.next_batch(batch_size)
    else:
        for _ in range(batch_limit):
            yield mnist.test.next_batch(batch_size)


def get_mnist_batch_xs_generator(is_train, batch_size, batch_limit=100):
    for x, _ in get_mnist_batch_generator(is_train, batch_size, batch_limit):
        yield x


def main():
    sess = tf.Session()
    sda = SDAutoencoder(dims=[784, 400, 200, 80],
                        activations=["sigmoid", "sigmoid", "sigmoid"],
                        sess=sess,
                        noise=0.25,
                        loss="cross-entropy",
                        lr=0.0001)

    mnist_train_gen_f = lambda: get_mnist_batch_xs_generator(True, batch_size=100, batch_limit=5000)

    sda.pretrain_network_gen(mnist_train_gen_f)
    trained_parameters = sda.finetune_parameters_gen(get_mnist_batch_generator(True, batch_size=100, batch_limit=50000),
                                                     output_dim=10)
    transformed_filepath = "../data/mnist_test_transformed.csv"
    test_ys_filepath = "../data/mnist_test_ys.csv"
    output_filepath = "../data/mnist_pred_ys.csv"

    sda.write_encoded_input_with_ys(transformed_filepath, test_ys_filepath,
                                    get_mnist_batch_generator(False, batch_size=100, batch_limit=100))
    sess.close()

    test_model(parameters_dict=trained_parameters,
               input_dim=80,
               output_dim=10,
               x_test_filepath=transformed_filepath,
               y_test_filepath=test_ys_filepath,
               output_filepath=output_filepath)

if __name__ == "__main__":
    main()
