"""Neural network models."""

import tensorflow as tf
import numpy as np

from aes import datasets
from aes.datasets import hp  # global hyperparameters

from aes.decorators import define_scope  # Danijar's decorators

class Model:
    """TODO"""
    def __init__(self, hp, essay, score):
        """TODO"""
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
        """TODO (Evaluate), using QWK"""
        pass

    def infer(self):
        # TODO
        pass


# def main():
#     print("training")
#     aes_sess = AesSession(hp, train_set)
#     aes_sess.initialize_variables()
#     aes_sess.train()
