"""
a random forest classifier
with muilti-GPU utilization

Tiffany.Fu

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import datasets, metrics, cross_validation
import tensorflow as tf
from tensorflow.contrib import skflow



import tensorflow as tf


class TensorForestTrainer (tf.test.TestCase):

  def Classification(self):
    """classification using matrix data as input."""
    hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_trees=300, max_nodes=1000, num_classes=2, num_features=4)
    classifier = tf.contrib.learn.TensorForestEstimator(hparams)


    classifier.fit(x=, y=, steps=100)
    classifier.evaluate(x=, y=, steps=10)



if __name__ == '__main__':
  tf.test.main()
