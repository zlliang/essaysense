"""Neural network models."""

import tensorflow as tf
from tensorflow.contrib import rnn as tfrnn
# import numpy as np

from aes import datasets
from aes.configs import hp, paths  # global hyperparameters
from aes.qwk import qwk # quadratic weighted kappa

from aes.decorators import define_scope  # Danijar's decorators

class Model:
    """TODO"""
    def __init__(self, hp, essays, scores):
        """TODO"""
        self.essays = essays
        self.scores = scores
        self.predictions
        self.optimize
        self.error

    @define_scope
    def predictions(self):
        """TODO"""
        essays = self.essays
        input_layer = tf.reshape(essays, [-1, hp.lstm_e_len, hp.w_dim])
        conv1 = tf.layers.conv1d(
            inputs=input_layer,
            filters=hp.lstm_convunits_size,
            kernel_size=hp.w_window_len,
            padding="same",
            activation=None)
        bn1 = tf.layers.batch_normalization(inputs=conv1)
        activated1 = tf.nn.relu(bn1)
        lstm_cell = tfrnn.BasicLSTMCell(num_units=hp.lstm_hidden_size)
        lstm_cell = tfrnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=hp.lstm_dropout_keep_prob)
        init_state = lstm_cell.zero_state(hp.batch_size, dtype=tf.float32)
        lstm_output, _ = tf.nn.dynamic_rnn(lstm_cell, activated1, dtype=tf.float32)
        mot = tf.reduce_mean(lstm_output, axis=1, name="EssayRepresentation")
        linear = tf.layers.dense(inputs=mot, units=1, activation=tf.nn.sigmoid)
        preds = tf.reshape(linear, [-1], name="PREDS")
        return preds

    @define_scope
    def optimize(self):
        """TODO"""
        # TODO: experiment: dense layer
        optimizer = tf.train.AdamOptimizer(hp.learning_rate)
        return optimizer.minimize(loss=self.error)

    @define_scope
    def error(self):
        """TODO"""
        # for now: MSE! TODO!!!
        return tf.losses.mean_squared_error(self.predictions, self.scores)

class AesSession:
    """TODO"""
    def __init__(self, hp, train_set):
        """TODO"""
        self.train_set = train_set
        self.sess = tf.Session()
        self.essays = tf.placeholder(tf.float32, [None, hp.e_len, hp.s_len, hp.w_dim])
        self.scores = tf.placeholder(tf.float32, [None])
        self.model = Model(hp, self.essays, self.scores)

    def initialize_variables(self):
        # try:
        #     raise NotImplementedError  # TODO: read checkpoints
        # except:
        #     sess.run(tf.global_variables_initializer())
        #     raise NotImplementedError  # TODO: train a new model?
        self.sess.run(tf.global_variables_initializer())

    def close(self):
        self.sess.close()

    def train(self):
        # TODO
        for i in range(100):
            essays_batch, scores_batch = self.train_set.next_batch(hp.batch_size)
            error = self.sess.run(self.model.error, {self.essays: essays_batch, self.scores: scores_batch})
            print('Error: %.3f' % error)
            self.sess.run(self.model.optimize, {self.essays: essays_batch, self.scores: scores_batch})

    def evaluate(self):
        """TODO (Evaluate), using QWK"""
        pass

    def infer(self):
        # TODO
        pass


def main():
    print("training")
    aes_sess = AesSession(hp, datasets.lstm_train_set)
    aes_sess.initialize_variables()
    aes_sess.train()
    aes_sess.close()
