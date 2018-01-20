from __future__ import division, absolute_import, print_function

import tensorflow as tf
import numpy as np
import nltk
import re

import datasets

# hyperparameters loading
hp = datasets.hp

source = tf.placeholder(shape=[None, hp.e_len, hp.s_len, hp.w_dim],
                        dtype=tf.float32)  # [batch, in_height, in_width, in_channels]
score = tf.placeholder(shape=[None], dtype=tf.float32)
state = tf.placeholder(dtype=tf.string)

log_dir = './summary'


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)


# conv1
with tf.name_scope('conv1') as scope:
    # [filter_height, filter_width, in_channels, out_channels]
    kernel = tf.Variable(tf.truncated_normal([1, hp.w_window_len, hp.w_dim, hp.w_convunits_size], stddev=hp.stddev),
                         dtype=tf.float32, name='weights')
    biases = tf.Variable(tf.truncated_normal([hp.w_convunits_size], stddev=hp.stddev), dtype=tf.float32, name='biases')
    conv = tf.nn.conv2d(source, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
    _variable_summaries(kernel)
    _variable_summaries(biases)
    if state == "Test":
        tf.summary.histogram('conv1', conv1)

# pool1
pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, hp.s_len, 1], strides=[1, 1, 1, 1],
                       padding='VALID', name='pool1')  # OUTPUT_DIM: [batch_size, hp.e_len, 1, hp.w_convunits_size]

# TODO: LRN layer

# conv2
with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(
        tf.truncated_normal([hp.s_window_len, 1, hp.w_convunits_size, hp.s_convunits_size], stddev=hp.stddev),
        dtype=tf.float32, name='weights')
    biases = tf.Variable(tf.truncated_normal([hp.s_convunits_size], stddev=hp.stddev), dtype=tf.float32, name='biases')
    conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
    _variable_summaries(kernel)
    _variable_summaries(biases)
    if state == "Test":
        tf.summary.histogram('conv2', conv2)
# pool2
pool2 = tf.nn.max_pool(conv2, ksize=[1, hp.e_len, 1, 1], strides=[1, 1, 1, 1],
                       padding='VALID', name='pool2')  # OUTPUT_DIM: [batch_size, 1, 1, hp.s_convunits_size]

# TODO: LRN layer

# dropout
pool2_drop = tf.nn.dropout(pool2, hp.dropout_keep_prob)

# local3
with tf.name_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 1
    for d in pool2.get_shape()[1:].as_list():
        dim *= d
    if state == "Train":
        reshape = tf.reshape(pool2_drop, [-1, dim])
    else:
        reshape = tf.reshape(pool2, [-1, dim])
    weights = tf.Variable(tf.truncated_normal([dim, hp.hidden_size], stddev=hp.stddev), dtype=tf.float32,
                          name='weights')
    biases = tf.Variable(tf.truncated_normal([hp.hidden_size], stddev=hp.stddev), dtype=tf.float32, name='biases')
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope)  # OUTPUT_DIM: [batch_size, hp.hidden_size]
    _variable_summaries(weights)
    _variable_summaries(biases)
    if state == "Test":
        tf.summary.histogram('local3', local3)

# softmax
with tf.name_scope('output') as scope:
    weights = tf.Variable(tf.truncated_normal([hp.hidden_size, 1], stddev=hp.stddev), dtype=tf.float32, name='weights')
    biases = tf.Variable(tf.truncated_normal([1], stddev=hp.stddev), dtype=tf.float32, name='biases')
    multi = tf.matmul(local3, weights) + biases
    predictions = tf.nn.sigmoid(multi, name='predictions')  # OUTPUT_DIM: [batch_size, 1]
    if state == "Test":
        tf.summary.histogram('predictions', predictions)
mse = tf.reduce_mean(tf.square(predictions - score))
if state == "Test":
    tf.summary.scalar('loss', mse)
merged = tf.summary.merge_all()

# -----------------------------------------------------------------------------

#
# #-----selfcheck
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # c = sess.run([pool1, pool2, local3, predictions, mse], feed_dict={source: datasets.next_batch(hp.batch_size)[0],
#     #                                                                   score: datasets.next_batch(hp.batch_size)[1]})
#     # print(c[0].shape)
#     # print(c[1].shape)
#     # print(c[2].shape)
#     # print(c[3].shape)
#     # print(c[4])
#     c = sess.run(predictions, feed_dict={source: datasets.train_set.next_batch(hp.batch_size)[0],
#                                    score: datasets.train_set.next_batch(hp.batch_size)[1]})
#     print(c)

# -----training
saver = tf.train.Saver()
isTrained = False
train_steps = 10000
checkpoint_dir = ''
train_set0 = datasets.train_set.next_batch(100)
train_error = []


def feed_dict(st):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if st == "Train":
        xs, ys = datasets.train_set.next_batch(hp.batch_size)
    elif st == "Test":
        xs, ys = datasets.test_set.all()
    return {source: xs, score: ys, state: st}


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    optimizer = tf.train.GradientDescentOptimizer(hp.learning_rate)
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    if isTrained:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        for iter in range(train_steps):
            summary, _ = sess.run([merged, optimizer.minimize(mse)], feed_dict=feed_dict("Train"))
            train_writer.add_summary(summary, iter)
            if (iter + 1) % 10 == 0:
                summary, test_error = sess.run([merged, mse], feed_dict=feed_dict("Test"))
                test_writer.add_summary(summary, iter)
                print("Iteration:{}\tTest_error:{:.6f}".format((iter + 1), test_error))
            if (iter + 1) % 100 == 0:
                saver.save(sess, './model.ckpt')
