#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_data_dir', '../mnist',
                           """Data where MNIST data is stored""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                          """Initial learning rate""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images in a batch""")
tf.app.flags.DEFINE_integer('n_hidden', 300,
                            """Number of hidden units""")
tf.app.flags.DEFINE_float('lambda_l1', 0.0,
                          """L1 regularisation strength""")
tf.app.flags.DEFINE_float('lambda_l2', 0.0,
                          """L2 regularisation strength""")
tf.app.flags.DEFINE_integer('max_iter', 10000,
                            """Maximum number of training iterations""")
tf.app.flags.DEFINE_string('run', 'run1',
                           """Subdirectory name for log files""")
N_CLASSES = 10


def two_layer_network(n_hidden, batch_size, lambda_l1, lambda_l2):
    images = tf.placeholder(tf.float32, shape=(batch_size,
                                               mnist.IMAGE_PIXELS),
                            name='images')
    labels = tf.placeholder(tf.int32, shape=(batch_size), name='labels')
    stddev_hidden = 1.0/math.sqrt(float(mnist.IMAGE_PIXELS))
    with tf.name_scope('hidden'):
        W0 = tf.Variable(tf.truncated_normal([mnist.IMAGE_PIXELS, n_hidden],
                                             stddev=stddev_hidden,
                                             dtype=tf.float32), name='W0')
        b0 = tf.Variable(tf.zeros([n_hidden]), tf.float32, name='b0')
        act0 = tf.matmul(images, W0) + b0
        hidden = tf.nn.relu(act0)
        tf.summary.histogram('act0', act0)
        tf.summary.histogram('W0', W0)
        W0_min = tf.reduce_min(W0)
        W0_max = tf.reduce_max(W0)
        W0p = (W0 - W0_min) / (W0_max - W0_min)
        I0 = tf.reshape(W0p, [28, 28, 1, n_hidden])
        tf.summary.image('filters', tf.transpose(I0, [3, 0, 1, 2]),
                         max_outputs=32)
    stddev_output = 1.0/math.sqrt(float(n_hidden))
    with tf.name_scope('output'):
        W1 = tf.Variable(tf.truncated_normal([n_hidden, N_CLASSES],
                                             stddev=stddev_output,
                                             dtype=tf.float32), name='W1')
        b1 = tf.Variable(tf.zeros([N_CLASSES]), tf.float32, name='b1')
        logits = tf.matmul(hidden, W1) + b1
        tf.summary.histogram('W1', W1)
    with tf.name_scope('loss'):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        data_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=lambda_l1, scope=None)
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=lambda_l2, scope=None)
        l1_loss = tf.contrib.layers.apply_regularization(l1_regularizer, W0) + tf.contrib.layers.apply_regularization(l1_regularizer, W1)
        l2_loss = tf.contrib.layers.apply_regularization(l2_regularizer, W0) + tf.contrib.layers.apply_regularization(l2_regularizer, W1)
        loss = data_loss + l1_loss + l2_loss
        tf.summary.scalar('data loss', data_loss)
        tf.summary.scalar('L1 loss', l1_loss)
        tf.summary.scalar('L2 loss', l2_loss)
        tf.summary.scalar('combined loss', loss)
    with tf.name_scope('accuracy'):
        correct = tf.nn.in_top_k(logits, labels, 1)
        n_correct = tf.to_float(tf.reduce_sum(tf.cast(correct, tf.int32)))
        accuracy = n_correct*100.0/tf.to_float(batch_size)
        tf.summary.scalar('accuracy', accuracy)
    return images, labels, logits, loss, accuracy


def main(argv=None):
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir)
    with tf.Graph().as_default():
        images, labels, logits, loss, acc = two_layer_network(FLAGS.n_hidden,
                                                              FLAGS.batch_size,
                                                              FLAGS.lambda_l1,
                                                              FLAGS.lambda_l2)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        init = tf.global_variables_initializer()
        summary = tf.summary.merge_all()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter('./logs/' + FLAGS.run, sess.graph)
        sess.run(init)

        for it in xrange(FLAGS.max_iter):
            image_b, label_b = data_sets.train.next_batch(FLAGS.batch_size)
            _, lv, av = sess.run([train_op, loss, acc],
                                 feed_dict={images: image_b,
                                            labels: label_b})
            if (it % 50 == 0):
                summary_str = sess.run(summary, feed_dict={images: image_b,
                                                           labels: label_b})
                summary_writer.add_summary(summary_str, it)
                summary_writer.flush()
            if (it % 500 == 0):
                msg = 'Iteration {:5d}: loss is {:7.4f}, accuracy is {:6.2f}%'
                print(msg.format(it, lv, av))
        print('Training completed')
        avg_accuracy = 0.0
        n_evals = data_sets.test.num_examples // FLAGS.batch_size
        for i in xrange(n_evals):
            image_b, label_b = data_sets.test.next_batch(FLAGS.batch_size)
            _, lv, av = sess.run([train_op, loss, acc],
                                 feed_dict={images: image_b,
                                            labels: label_b})
            avg_accuracy += av
        avg_accuracy /= n_evals
        print('Test accuracy is {:.2f}%'.format(avg_accuracy))


if __name__ == '__main__':
    tf.app.run()
