"""Neural network models."""

import tensorflow as tf
import numpy as np
import nltk

from aes import datasets

from aes.decorators import define_scope

# hyperparameters loading
hp = datasets.hp

class Model:
    """TODO"""
    def __init__(self, hp, essay, score):
        """TODO"""
        self.hp = hp # TODO: private!
        self.essay = essay
        self.score = score
        self.prediction
        self.optimize
        self.loss

    @define_scope
    def prediction(self):
        """TODO"""
        return result

    @define_scope
    def optimize(self):
        """TODO"""
        # TODO: experiment: dense layer
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        return optimizer.minimize(self.loss)

    @define_scope
    def loss(self):
        """TODO"""
        # for now: MSE! TODO!!!
        return tf.square(self.prediction - self.score)

class AesSession:
    """TODO"""
    def __init__(self, hp, train_set):
        """TODO"""
        self.train_set = train_set
        self.sess = tf.Session()
        self.essay = tf.placeholder(tf.float32, [hp.e_len, hp.s_len, hp.w_dim])
        self.score = tf.placeholder(tf.float32, [1])
        self.model = Model(hp, self.essay, self.score)

    def initialize_variables(self):
        # try:
        #     raise NotImplementedError  # TODO: read checkpoints
        # except:
        #     sess.run(tf.global_variables_initializer())
        #     raise NotImplementedError  # TODO: train a new model?
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        # TODO
        for i in range(30):
            essay = train_set[i]["essay"]
            score = train_set[i]["score"]
            error = self.sess.run(self.model.loss, {self.essay: essay, self.score: score})
            print('Error: %.3f' % error)
            self.sess.run(self.model.optimize, {self.essay: essay, self.score: score})

    def evaluate(self):
        # TODO
        pass

    def infer(self):
        # TODO
        pass


def main():
    print("training")
    aes_sess = AesSession(hp, train_set)
    aes_sess.initialize_variables()
    aes_sess.train()










# def max_avg_pooling(data, height):
#     maxpool = tf.nn.max_pool(data, ksize=[1, height, 1, 1], strides=[1, height, 1, 1], padding='SAME')
#     avgpool = tf.nn.avg_pool(data, ksize=[1, height, 1, 1], strides=[1, height, 1, 1], padding='SAME')
#     return tf.concat([maxpool, avgpool], 2)
#
#
# def max_pooling(data, height):
#     return tf.nn.max_pool(data, ksize=[1, height, 1, 1], strides=[1, height, 1, 1], padding='SAME')
#
#
# source = []
# layer0_embedding = []
# layer1_conv = []
# layer1_activated = []
# layer1_pool = []
# layer1_filt = tf.Variable(tf.random_normal([DIM, WORD_WINDOW_SIZE, 1, 1]),
#                           dtype=tf.float32, name='word-filter')
# layer1_bias = tf.Variable(tf.random_normal([1]))
# for i in range(SENTENCE_SIZE):
#     source.append(tf.placeholder(dtype=tf.float32))
#     layer0_embedding.append(tf.reshape(source[i], [-1, WORD_SIZE, WORD_CONV_UNITS, 1]))
#     layer1_conv.append(tf.nn.bias_add(tf.nn.conv2d(layer0_embedding[i],
#                                                    layer1_filt,
#                                                    strides=[1, 1, 1, 1],
#                                                    padding='SAME'),
#                                       layer1_bias))
#     layer1_activated.append(tf.nn.relu(layer1_conv[i]))
#     layer1_pool.append(max_pooling(layer1_activated[i], WORD_SIZE))
#
# layer2_source = tf.reshape(tf.convert_to_tensor(layer1_pool), [1, SENTENCE_SIZE, DIM, 1])
# layer2_filt = tf.Variable(tf.random_normal([DIM, SENTENCE_WINDOW_SIZE, 1, 1]),
#                           dtype=tf.float32, name='sentence-filter')
# layer2_bias = tf.Variable(tf.random_normal([1]))
# layer2_conv = tf.nn.bias_add(tf.nn.conv2d(layer2_source,
#                                           layer2_filt,
#                                           strides=[1, 1, 1, 1],
#                                           padding='SAME'),
#                              layer2_bias)
# layer2_activated = tf.nn.relu(layer2_conv)
# layer2_pool = max_pooling(layer2_activated, SENTENCE_SIZE)
#
# layer3_source = tf.reshape(layer2_pool, [1, DIM])
# layer3_weight = tf.Variable(tf.random_normal([DIM, SENTENCE_SIZE]))
# layer3_bias = tf.Variable(tf.random_normal([SENTENCE_SIZE]))
# layer3_result = tf.nn.relu(tf.matmul(layer3_source, layer3_weight) + layer3_bias)
#
# layer4_weight = tf.Variable(tf.random_normal([SENTENCE_SIZE, 1]))
# layer4_bias = tf.Variable(tf.random_normal([1]))
# prediction = tf.nn.sigmoid(tf.matmul(layer3_result, layer4_weight) + layer4_bias)
#
# human_score = tf.placeholder(dtype=tf.float32)
# mse = tf.square(prediction - human_score)
#
#
# # -----------------------------------------------------------------------------
#
# def get_essay_feed(index):
#     asap_item = asap_set1[index]
#     embedded = get_essay_embedding(asap_item['essay'])
#     fed = {}
#     for j in range(SENTENCE_SIZE):
#         fed[source[j]] = embedded[j]
#     fed[human_score] = (int(asap_item['score']) - 2) / 10
#     return fed
#
#
# # # -----selfcheck
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     c = sess.run(prediction, feed_dict=get_essay_feed(0))
# #     print(c)
# #     print(c[0].shape)
# #     print(len(c))
#
# saver = tf.train.Saver()
#
# isTrained = False
# train_steps = 1000
# checkpoint_steps = 100
# checkpoint_dir = ''
# ESSAY_SIZE = 1783
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
#
#     if isTrained:
#         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#         if ckpt and ckpt.model_checkpoint_path:
#             saver.restore(sess, ckpt.model_checkpoint_path)
#     else:
#         for iter in range(train_steps):
#             essay_id = np.random.randint(1, ESSAY_SIZE)
#             sess.run(optimizer.minimize(mse), feed_dict=get_essay_feed(essay_id))
#             if (iter + 1) % 10 == 0:
#                 print(iter + 1)
#                 if (iter + 1) % checkpoint_steps == 0:
#                     saver.save(sess, './model.ckpt')
#
#     # Test
#     for test_iter in range(10):
#         test_index = test_iter + 1000
#         fed_test = get_essay_feed(test_index)
#         test_prediction = prediction.eval(feed_dict=fed_test)
#         test_mse = mse.eval(feed_dict=fed_test)
#         print(test_prediction, test_mse)
