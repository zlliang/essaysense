import tensorflow as tf
from tensorflow.contrib import rnn as tfrnn

from aes.configs import paths
from aes.modeltemplate import Model

class SentenceLevelCnn(Model):
    """Hierarchical CNN model for AES task.

    One convolutional layer with batch normalization and max pooling converts
    word representations to sentence embeddings, and another convolutional layer
    with batch normalization and max pooling converts sentence embeddings to
    essay representations. Then two full-connected layers convert essay repre-
    sentations to a single score.

    Network Topology:
        [essay] -> conv1 -> bn1 -> pool1 -> conv2 -> bn2 -> pool2
                -> dense1 -> dense2 -> [prediction score]

    Reference:
        This model basically implemented based on the article "Automatic
        Features for Essay Scoring -- An Empirical Study"
        (Fei Dong and Yue Zhang 2016).
    """
    def __init__(self, hyperparameters, train_set, test_set, domain_id=1):
        super().__init__(hyperparameters, train_set, test_set, domain_id)
        self.name = "prompt-" + self.domain_id + "-sentence-level-cnn"

    def define_graph(self):
        essays = tf.placeholder(tf.float32, [None, self.hp.e_len,
                                             self.hp.s_len, self.hp.w_dim])
        scores = tf.placeholder(tf.float32, [None])

        # Convolutional layer 1
        conv1 = tf.layers.conv2d(
            inputs=essays,
            filters=self.hp.w_convunits_size,
            kernel_size=[1, self.hp.w_window_len],
            padding="same",
            activation=None)
        bn1 = tf.layers.batch_normalization(inputs=conv1)
        activated1 = tf.nn.relu(bn1)
        pool1 = tf.layers.max_pooling2d(inputs=activated1,
                                        pool_size=[1, self.hp.s_len], strides=1)
        # Convolutional layer 2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=self.hp.s_convunits_size,
            kernel_size=[self.hp.s_window_len, 1],
            padding="same",
            activation=None)
        bn2 = tf.layers.batch_normalization(inputs=conv2)
        activated2 = tf.nn.relu(bn2)
        pool2 = tf.layers.max_pooling2d(inputs=activated2,
                                        pool_size=[self.hp.e_len, 1], strides=1)
        pool2_flat = tf.reshape(pool2, [-1, self.hp.s_convunits_size])

        # Dense layers
        dense1 = tf.layers.dense(inputs=pool2_flat,
                                 units=self.hp.hidden_size,
                                 activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1,
                                 units=1, activation=tf.nn.sigmoid)

        # Prediction and Loss
        preds = tf.reshape(dense2, [-1])
        tf.summary.histogram('preds', preds)
        loss = tf.losses.mean_squared_error(scores, preds)
        tf.summary.scalar('loss', loss)
        merged_summary = tf.summary.merge_all()

        return (essays,
                scores,
                merged_summary,
                loss,
                preds)

class SentenceLevelCnnLstmWithAttention(Model):
    """Hierarchical CNN-LSTM model for AES task.

    A convolutional layer with attention pooling technique converts word repre-
    sentations to sentence embeddings, and a LSTM layer also with attention
    pooling converts sentence embeddings to essay representations. Then one
    dense activation layer converts essay representations to a single score.

    Network Topology:
        [essay] -> conv1 -> att_pool1 -> lstm -> att_pool2
                -> dense -> [prediction score]

    Reference:
        This model basically implemented based on the article "Attention-based
        Recurrent Convolutional Neural Network for Automatic Essay Scoring"
        (Fei Dong et al., 2017).
    """
    def __init__(self, hyperparameters, train_set, test_set, domain_id=1):
        super().__init__(hyperparameters, train_set, test_set, domain_id)
        self.name = "prompt-" + self.domain_id \
                    + "-sentence-level-cnn-lstm-with-attention"

    def define_graph(self):
        essays = tf.placeholder(tf.float32, [None, self.hp.e_len,
                                             self.hp.s_len, self.hp.w_dim])
        scores = tf.placeholder(tf.float32, [None])

        # Convolutional layer
        conv1 = tf.layers.conv2d(
            inputs=essays,
            filters=self.hp.cnn_lstm_convunits_size,
            kernel_size=[1, self.hp.w_window_len],
            padding="same",
            activation=tf.nn.relu)

        # Attention pooling
        att1_mat = tf.Variable(
            tf.truncated_normal([self.hp.cnn_lstm_convunits_size,
                                 self.hp.cnn_lstm_convunits_size]),
            dtype=tf.float32)
        att1_bias = tf.Variable(
            tf.truncated_normal([1, 1, 1,
                                 self.hp.cnn_lstm_convunits_size]),
            dtype=tf.float32)
        att1_weight = tf.tensordot(conv1, att1_mat, axes=[3, 0]) + att1_bias
        att1_weight = tf.nn.tanh(att1_weight)
        att1_vec = tf.Variable(
            tf.truncated_normal([self.hp.cnn_lstm_convunits_size, 1]),
            dtype=tf.float32)
        att1_weight = tf.tensordot(att1_weight, att1_vec, axes=[3, 0])
        att1_weight = tf.nn.softmax(att1_weight, dim=2)
        att1_output = att1_weight * conv1
        att1_output = tf.reduce_sum(att1_output, axis=2)

        # Long Short-Term Memory layer
        lstm_cell = tfrnn.BasicLSTMCell(
            num_units=self.hp.cnn_lstm_att_pool_size)
        lstm_cell = tfrnn.DropoutWrapper(
            cell=lstm_cell,
            output_keep_prob=self.hp.dropout_keep_prob)
        init_state = lstm_cell.zero_state(self.hp.batch_size, dtype=tf.float32)
        lstm, _ = tf.nn.dynamic_rnn(lstm_cell, att1_output, dtype=tf.float32)

        # Attention pooling
        att2_mat = tf.Variable(
            tf.truncated_normal([self.hp.cnn_lstm_att_pool_size,
                                 self.hp.cnn_lstm_att_pool_size]),
            dtype=tf.float32)
        att2_bias = tf.Variable(
            tf.truncated_normal([1, 1,
                                 self.hp.cnn_lstm_att_pool_size]),
            dtype=tf.float32)
        att2_weight = tf.tensordot(lstm, att2_mat, axes=[2, 0])
        att2_weight = tf.nn.tanh(att2_weight)
        att2_vec = tf.Variable(
            tf.truncated_normal([self.hp.cnn_lstm_att_pool_size,
                                 1]),
            dtype=tf.float32)
        att2_weight = tf.tensordot(att2_weight, att2_vec, axes=[2, 0])
        att2_weight = tf.nn.softmax(att2_weight, dim=1)
        att2_output = att2_weight * lstm
        att2_output = tf.reduce_sum(att2_output, axis=1)

        # Dense layer
        dense = tf.layers.dense(inputs=att2_output, units=1,
                                activation=tf.nn.sigmoid)

        # Prediction and Loss
        preds = tf.reshape(dense, [-1])
        tf.summary.histogram('preds', preds)
        loss = tf.losses.mean_squared_error(scores, preds)
        tf.summary.scalar('loss', loss)
        merged_summary = tf.summary.merge_all()

        return (essays,
                scores,
                merged_summary,
                loss,
                preds)


class DocumentLevelLstmWithMotPooling(Model):
    """An LSTM model for AES task.

    This model treat a piece of essay as a single sequence of words. LSTM cells
    accept the sequence and output a processed sequence. Then mean over time
    (MoT) pooling is performed which converts the feed tensor into an essay
    representation vector. Then a dense layer converts the vector to an
    appropriate score of the essay.

    Network Topology:
        [essay] -> lstm -> mot pooling -> dense -> [prediction score]

    Reference:
        This model basically implemented based on the article "A neural
        Approach to Automated Essay Scoring"
        (Taghipour and Ng, 2016).
    """
    def __init__(self, hyperparameters, train_set, test_set, domain_id=1):
        super().__init__(hyperparameters, train_set, test_set, domain_id)
        self.name = "prompt-" + self.domain_id \
                    + "-document-level-lstm-with-mot-pooling"

    def define_graph(self):
        essays = tf.placeholder(tf.float32, [None, self.hp.d_e_len,
                                             self.hp.w_dim])
        scores = tf.placeholder(tf.float32, [None])

        # Long Short-Term Memory layer
        lstm_cell = tfrnn.BasicLSTMCell(num_units=self.hp.lstm_hidden_size)
        lstm_cell = tfrnn.DropoutWrapper(
            cell=lstm_cell,
            output_keep_prob=self.hp.dropout_keep_prob)
        init_state = lstm_cell.zero_state(self.hp.batch_size, dtype=tf.float32)
        lstm, _ = tf.nn.dynamic_rnn(lstm_cell, essays, dtype=tf.float32)

        # Mean over Time pooling
        mot = tf.reduce_mean(lstm, axis=1)

        # Dense layer
        dense = tf.layers.dense(inputs=mot, units=1, activation=tf.nn.sigmoid)

        # Prediction and Loss
        preds = tf.reshape(dense, [-1])
        tf.summary.histogram('preds', preds)
        loss = tf.losses.mean_squared_error(scores, preds)
        tf.summary.scalar('loss', loss)
        merged_summary = tf.summary.merge_all()

        return (essays,
                scores,
                merged_summary,
                loss,
                preds)
