#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
tf.app.flags.DEFINE_integer('max_iter', 10000,
                            """Maximum number of training iterations""")
tf.app.flags.DEFINE_string('run', 'run1',
                           """Subdirectory name for log files""")
N_CLASSES = 10


def conv_network(batch_size):
    images = tf.placeholder(tf.float32, shape=(batch_size,
                                               mnist.IMAGE_PIXELS),
                            name='images')
    labels = tf.placeholder(tf.int32, shape=(batch_size), name='labels')
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
    input_layer = tf.reshape(images, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=is_training)
    logits = tf.layers.dense(inputs=dropout, units=10)
    with tf.name_scope('loss'):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.summary.scalar('loss', loss)
    with tf.name_scope('accuracy'):
        correct = tf.nn.in_top_k(logits, labels, 1)
        n_correct = tf.to_float(tf.reduce_sum(tf.cast(correct, tf.int32)))
        accuracy = n_correct*100.0/tf.to_float(batch_size)
        tf.summary.scalar('accuracy', accuracy)
    return images, labels, is_training, logits, loss, accuracy


def main(argv=None):
    modelpath = "/tmp/model.ckpt"
    
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir)
    g1 = tf.Graph()
    with g1.as_default():
        images, labels, is_training, logits, loss, acc = conv_network(
            FLAGS.batch_size)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        init = tf.global_variables_initializer()
        with tf.variable_scope('conv2d') as scope:
            scope.reuse_variables()
            kernel = tf.get_variable('kernel')
            k_min = tf.reduce_min(kernel)
            k_max = tf.reduce_max(kernel)
            I0 = (kernel - k_min) / (k_max - k_min)
            tf.summary.image('filters', tf.transpose(I0, [3, 0, 1, 2]),
                             max_outputs=32)
        summary = tf.summary.merge_all()
        
        saver = tf.train.Saver() # included Saver
        
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter('./logs/' + FLAGS.run,
                                               sess.graph)                                               
    
        sess.run(init)

        for it in xrange(FLAGS.max_iter):
            image_b, label_b = data_sets.train.next_batch(FLAGS.batch_size)
            _, lv, av = sess.run([train_op, loss, acc],
                                 feed_dict={images: image_b,
                                            labels: label_b,
                                            is_training: True})
            if (it % 50 == 0):
                summary_str = sess.run(summary, feed_dict={images: image_b,
                                                           labels: label_b,
                                                           is_training: True})
                summary_writer.add_summary(summary_str, it)
                summary_writer.flush()
            if (it % 500 == 0):
                msg = 'Iteration {:5d}: loss is {:7.4f}, accuracy is {:6.2f}%'
                print(msg.format(it, lv, av))
        print('Training completed')
        
        save_path = saver.save(sess, modelpath) # Save parameters to disk
        print("Model saved in file: %s" % save_path)
        
    g2 = tf.Graph()
    with g2.as_default():
        images, labels, is_training, logits, loss, acc = conv_network(
            FLAGS.batch_size)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        init = tf.global_variables_initializer()
        with tf.variable_scope('conv2d') as scope:
            scope.reuse_variables()
            kernel = tf.get_variable('kernel')
            k_min = tf.reduce_min(kernel)
            k_max = tf.reduce_max(kernel)
            I0 = (kernel - k_min) / (k_max - k_min)
            tf.summary.image('filters', tf.transpose(I0, [3, 0, 1, 2]),
                             max_outputs=32)
        summary = tf.summary.merge_all()
        
        summary_writer = tf.summary.FileWriter('./logs/' + FLAGS.run,
                                               sess.graph)
        sess2 = tf.Session()
        sess2.run(init)       
        saver = tf.train.Saver() # included Saver
        saver.restore(sess2, modelpath)

        
        
        
        avg_accuracy = 0.0
        n_evals = data_sets.test.num_examples // FLAGS.batch_size
        for i in xrange(n_evals):
            image_b, label_b = data_sets.test.next_batch(FLAGS.batch_size)
            _, lv, av = sess2.run([train_op, loss, acc],
                                 feed_dict={images: image_b,
                                            labels: label_b,
                                            is_training: False})
            avg_accuracy += av
        avg_accuracy /= n_evals
        print('Test accuracy is {:.2f}%'.format(avg_accuracy))


        

if __name__ == '__main__':
    tf.app.run()
