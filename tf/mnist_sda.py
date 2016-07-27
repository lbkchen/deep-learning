"""
Example testing SDA model on MNIST digits.
"""

from final_sda import get_batch_generator, SDAutoencoder
from softmax import test_model_gen, test_model
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def get_mnist_batch_generator(is_train, batch_size, batch_limit=100):
    if is_train:
        for _ in range(batch_limit):
            yield mnist.train.next_batch(batch_size)
    else:
        for _ in range(batch_limit):
            yield mnist.test.next_batch(batch_size)


# def get_mnist_batch_generators(batch_limit=100):
#     both_gen = get_mnist_batch_generator(batch_limit)
#     current_batch = next(both_gen)
#     x_retrieved = False
#     y_retrieved = False
#
#     def refresh():
#         nonlocal current_batch, x_retrieved, y_retrieved
#         if x_retrieved and y_retrieved:
#             current_batch = next(both_gen)
#             x_retrieved = False
#             y_retrieved = False
#
#     def get_x_batch_generator():
#         nonlocal x_retrieved
#         refresh()
#         x_retrieved = True
#         yield current_batch[0]
#
#     def get_y_batch_generator():
#         nonlocal y_retrieved
#         refresh()
#         y_retrieved = True
#         yield current_batch[1]
#
#     return get_x_batch_generator, get_y_batch_generator

def main():
    sess = tf.Session()
    sda = SDAutoencoder(dims=[784, 80, 80, 80],
                        activations=["sigmoid", "sigmoid", "sigmoid"],
                        sess=sess,
                        noise=0.05,
                        loss="cross-entropy")

    mnist_train_gen = get_mnist_batch_generator(True, batch_size=100, batch_limit=1000)

    sda.pretrain_network_gen(mnist_train_gen)
    trained_parameters = sda.finetune_parameters_gen(get_mnist_batch_generator(True, batch_size=100, batch_limit=2000),
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