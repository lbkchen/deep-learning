import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import time

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding="SAME")

def variable_summaries(var, name):
    """Attach some summaries to a tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def conv_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, fully_connected=False):
    """
    Makes a simple convolutional layer based on input and output dimensions.

    input_tensor: A tensor of the input data from the previous layer (of shape [a, b, c, d])

    Returns the pooled tensor after CONV -> ACT -> POOL
    """
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            weights = weight_variable([input_dim, output_dim]) if fully_connected else weight_variable([5, 5, input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope("biases"):
            bias = bias_variable([output_dim])
            variable_summaries(bias, layer_name + '/bias')
        if fully_connected:
            with tf.name_scope("fully_connected"):
                final = act(tf.matmul(input_tensor, weights) + bias)
                tf.histogram_summary(layer_name + '/fully_connected', final)
                return final
        else:
            with tf.name_scope("convolution"):
                convolution = act(conv2d(input_tensor, weights) + bias)
                tf.histogram_summary(layer_name + '/convolution', convolution)
                pooled = max_pool_2x2(convolution)
                return pooled

def dropout(tensor, keep_prob):
    with tf.name_scope('dropout'):
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        tensor_dropped = tf.nn.dropout(tensor, keep_prob)
    return tensor_dropped

def readout(input_tensor, input_dim, output_dim):
    weights = weight_variable([input_dim, output_dim])
    bias = bias_variable([output_dim])
    return tf.nn.softmax(tf.matmul(input_tensor, weights) + bias)

# def feed_dict(train, keep_prob):
#     if train:
#         xs, ys = mnist.train.next_batch(50)
#         k = 0.5
#     else:
#         xs, ys = mnist.test.images, mnist.test.labels
#         k = 1.0
#     return {x : xs, y_ : ys, keep_prob : k}

def main():
    start = time.time()
    sess = tf.InteractiveSession()
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

    with tf.name_scope("input_reshape"):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.image_summary('input', x_image, 10)

    layer1 = conv_layer(x_image, 1, 32, "layer1")
    layer2 = conv_layer(layer1, 32, 64, "layer2")
    layer2_flat = tf.reshape(layer2, [-1, 7 * 7 * 64])
    fc_layer = conv_layer(layer2_flat, 7 * 7 * 64, 1024, "layer_fc", fully_connected=True)

    keep_prob = tf.placeholder(tf.float32)
    fc_dropped = dropout(fc_layer, keep_prob)
    y_conv = readout(fc_dropped, 1024, 10)

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope("train_step"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)
#     merged = tf.merge_all_summaries()
# train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
#                                       sess.graph)
# test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
# tf.initialize_all_variables().run()
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

    tf.initialize_all_variables().run()
    # sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
 # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
 #        run_metadata = tf.RunMetadata()
 #        summary, _ = sess.run([merged, train_step],
 #                              feed_dict=feed_dict(True),
 #                              options=run_options,
 #                              run_metadata=run_metadata)
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict={x : batch[0], y_ : batch[1], keep_prob : 1.0})
            test_writer.add_summary(summary, i)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x : batch[0], y_ : batch[1], keep_prob : 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step],
            feed_dict={x : batch[0], y_ : batch[1], keep_prob : 1.0},
            options=run_options,
            run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
        if (i + 1) % 1000 == 0:
            model_name = 'deep_mnist_model'
            saver.save(sess, "run_data/" + model_name, global_step=i)
            print("saving model parameters to", model_name)
        train_step.run(feed_dict={x : batch[0], y_ : batch[1], keep_prob : 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={x : mnist.test.images, y_ : mnist.test.labels, keep_prob : 1.0}))
    end = time.time()
    print("Total time elapsed: " + str(end - start))

if __name__ == "__main__":
    main()
