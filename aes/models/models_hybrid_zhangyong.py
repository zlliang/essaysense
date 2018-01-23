import tensorflow as tf
from tensorflow.contrib import rnn as tfrnn

from aes.datasets import lstm_train_set, lstm_test_set
from aes.configs import paths, hp
from aes.qwk import qwk

def main():
    essays = tf.placeholder(tf.float32, [None, hp.lstm_e_len, hp.w_dim], name="essays")
    scores = tf.placeholder(tf.float32, [None], name="scores")

    input_layer = tf.reshape(essays, [-1, hp.lstm_e_len, hp.w_dim])

    conv = tf.layers.conv1d(
        inputs=input_layer,
        filters=hp.lstm_hidden_size,
        kernel_size=hp.w_window_len,
        padding="same",
        activation=tf.nn.relu)

    # lstm_cell = tfrnn.BasicLSTMCell(num_units=hp.lstm_hidden_size)
    # lstm_cell = tfrnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=hp.lstm_dropout_keep_prob)
    # init_state = lstm_cell.zero_state(hp.batch_size, dtype=tf.float32)
    #
    # lstm, _ = tf.nn.dynamic_rnn(lstm_cell, input_layer, dtype=tf.float32)
    # 
    # # attention pooling! TODO
    # norm1 = tf.sqrt(tf.reduce_sum(tf.square(conv), axis=2))
    # norm2 = tf.sqrt(tf.reduce_sum(tf.square(lstm), axis=2))
    # inner_prod = tf.reduce_sum(conv*lstm, axis=2)
    # sim = inner_prod / (norm1 * norm2 + 1e-4)
    # sim = tf.nn.softmax(sim, dim=1)
    # sim = tf.reshape(sim, [-1, hp.lstm_e_len, 1])
    # att_output = sim * conv
    # att_output = tf.reduce_sum(att_output, axis=1, keep_dims=True)

    linear = tf.layers.dense(inputs=conv, units=1, activation=tf.nn.sigmoid)

    preds = tf.reshape(linear, [-1], name="PREDS")
    tf.summary.histogram('preds', preds)

    loss = tf.losses.mean_squared_error(scores, preds)
    tf.summary.scalar('loss', loss)


    g_step = tf.Variable(0, dtype=tf.int64, trainable=False)

    learning_rate = tf.train.exponential_decay(hp.learning_rate, global_step=g_step, decay_steps=10, decay_rate=0.93)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss, global_step=g_step)

    # add_global = g_step.assign_add(1)

    test_essays, test_scores = lstm_test_set.all()

    # human_scores_tensor = tf.constant(test_scores, tf.float32)
    # tf.summary.histogram('human_scores', human_scores_tensor)
    merged = tf.summary.merge_all()



    saver = tf.train.Saver()
    # isTrained = False
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(paths.summary_train, sess.graph)
        # tsne! TODO
        # tfprojector.visualize_embeddings(summary_writer, projector_config)
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(paths.model_ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(5000):
            train_essays, train_scores = lstm_train_set.next_batch(hp.batch_size)
            sess.run(train_op, feed_dict={essays: train_essays, scores: train_scores})
            gstep, lrate, summary, test_loss, pred_scores = sess.run([g_step, learning_rate, merged, loss, preds], feed_dict={essays:test_essays, scores: test_scores})

            qwk_value = qwk(pred_scores, test_scores, 1)
            print("iter: {} \t loss: {:.6f} \t learning rate: {:.6f} \t QWK: {:.6f}".format(gstep, test_loss, lrate, qwk_value))
            summary_writer.add_summary(summary, gstep)

            # if gstep % 10 == 0:
                # saver.save(sess, paths.model, global_step=gstep)

if __name__ == "__main__":
    main()
