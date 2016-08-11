"""
Utility functions for SDA

Includes batch generation methods, and generator repeating/merging.

Ken Chen
"""

import random
import csv
import time
from math import ceil
from functools import wraps


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


def file_len(filename):
    """Returns the number of lines in a file."""
    i = 0
    with open(filename) as f:
        for i, line in enumerate(f):
            pass
    return i + 1


def get_batch_generator(filename, batch_size, repeat=0):
    """Generator that sequentially gets batches of batch_size x or y values
    from the given file.

    :param filename: A string of the file to write to.
    :param batch_size: An int: the number of lines to include in each batch.
    :param repeat: An int specifying the number of times to repeat going through
        the file. Repeat of 2 will return a generator that iterates through the
        full file three times before stopping iteration.
    :return: A generator.
    """
    assert repeat < 1000, "Recursion depth will be exceeded."
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

        # Catch any remainders in current data set
        if this_batch:
            yield this_batch

        print("Finished a batch iteration through %s" % filename)
        if repeat > 0:
            for item in get_batch_generator(filename, batch_size, repeat - 1):
                yield item


def get_random_batch_generator(batch_size, filename, paired_filename=None, repeat=0):
    """Given a csv file `filename` and a specified batch_size, returns a generator that randomly
    yields `batch_size` cases from the file at a time and repeats its entire set of rows for
    `repeat` number of times.

    Note: use only for smaller files, as this process will consume significant memory.

    :param batch_size: An int, the number of lines to include in each batch.
    :param filename: A string, the path to the file to be batched.
    :param paired_filename: A string (optional), the path to another file to be batched together
        with `filename`.
    :param repeat: An int, the number of times to repeat batching of the entire dataset.
    :return: If `paired_filename` is not None, returns a generator that yields corresponding tuples
        of batches from both datasets. If `paired_filename` is None, returns a generator that yields
        just batches from `filename`.
    """
    def batch_list(lst):
        return [lst[j*batch_size:(j+1)*batch_size] for j in range(int(ceil(len(lst) / batch_size)))]

    for _ in range(repeat + 1):
        with open(filename, "rt") as file:
            if paired_filename:
                with open(paired_filename, "rt") as paired:
                    paired = list(zip(list(csv.reader(file)), list(csv.reader(paired))))
                    random.shuffle(paired)
                    lines_0, lines_1 = list(zip(*paired))
                    lines_0, lines_1 = batch_list(lines_0), batch_list(lines_1)
                    for batch_0, batch_1 in zip(lines_0, lines_1):
                        yield batch_0, batch_1
            else:
                lines = list(csv.reader(file))
                random.shuffle(lines)
                lines = batch_list(lines)
                for batch in lines:
                    yield batch


def repeat_generator(f_gen, multiple=2):
    """Repeats a generator.

    :param f_gen: A function that when called with no arguments returns a generator
        to be repeated.
    :param multiple: The number of times the generator should be iterated through.
    :return: A generator that iterates through the original generator `multiple`
    number of times.
    """
    for _ in range(multiple):
        gen = f_gen()
        for item in gen:
            yield item


def merge_generators(gen_1, gen_2):
    """Returns a generator that yields combined tuples of the results of `gen_1` and `gen_2`."""
    for x, y in zip(gen_1, gen_2):
        yield x, y
