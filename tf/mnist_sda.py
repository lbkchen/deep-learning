"""
Example testing SDA model on MNIST digits.
"""

from final_sda import get_batch_generator, stopwatch, SDAutoencoder
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def get_mnist_batch_generator(batch_limit=100):
    for _ in range(batch_limit):
        yield mnist.train.next_batch(batch_limit)


def get_mnist_batch_generators(batch_limit=100):
    both_gen = get_mnist_batch_generator(batch_limit)
    current_batch = next(both_gen)
    x_retrieved = False
    y_retrieved = False

    def refresh():
        nonlocal current_batch, x_retrieved, y_retrieved
        if x_retrieved and y_retrieved:
            current_batch = next(both_gen)
            x_retrieved = False
            y_retrieved = False

    def get_x_batch_generator():
        nonlocal x_retrieved
        refresh()
        x_retrieved = True
        yield current_batch[0]

    def get_y_batch_generator():
        nonlocal y_retrieved
        refresh()
        y_retrieved = True
        yield current_batch[1]

    return get_x_batch_generator, get_y_batch_generator
