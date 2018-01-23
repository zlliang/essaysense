import tensorflow as tf
from tensorflow.contrib import rnn as tfrnn

from aes.datasets import train_set, test_set
from aes.configs import paths, hp
from aes.qwk import qwk

def main():
    essays = tf.placeholder(tf.float32, [None, hp.e_len, hp.s_len, hp.w_dim])
    scores = tf.placeholder(tf.float32, [None])

    input_layer = tf.reshape(essays, [-1, hp.e_len, hp.s_len, hp.w_dim])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=hp.lstm_sen_level_convunits_size,
        kernel_size=[1, hp.w_window_len],
        padding="same",
        activation=tf.nn.relu)

    # attention pooling 1 TODO
    att1_mat = tf.Variable(tf.truncated_normal([hp.lstm_sen_level_convunits_size, hp.lstm_sen_level_convunits_size]), dtype=tf.float32)
    att1_bias = tf.Variable(tf.truncated_normal([1, 1, 1, hp.lstm_sen_level_convunits_size]), dtype=tf.float32)
    att1_weight = tf.tensordot(conv1, att1_mat, axes=[3, 0]) + att1_bias
    att1_weight = tf.nn.tanh(att1_weight)
    att1_vec = tf.Variable(tf.truncated_normal([hp.lstm_sen_level_convunits_size, 1]), dtype=tf.float32)
    att1_weight = tf.tensordot(att1_weight, att1_vec, axes=[3, 0])
    att1_weight = tf.nn.softmax(att1_weight, dim=2)
    att1_output = att1_weight * conv1
    att1_output = tf.reduce_sum(att1_output, axis=2)

    # layer 2 -- CNN part
    conv2 = tf.layers.conv1d(
        inputs=att1_output,
        filters=hp.lstm_sen_level_att_pool_hidden_size,
        kernel_size=hp.s_window_len,
        padding="same",
        activation=tf.nn.relu)

    # layer 2 -- LSTM part
    lstm_cell = tfrnn.BasicLSTMCell(num_units=hp.lstm_sen_level_att_pool_hidden_size)
    lstm_cell = tfrnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=hp.lstm_dropout_keep_prob)
    init_state = lstm_cell.zero_state(hp.batch_size, dtype=tf.float32)
    lstm2, _ = tf.nn.dynamic_rnn(lstm_cell, att1_output, dtype=tf.float32)

    # layer 2 -- attention pooling (Yong Zhang's method)
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(conv2), axis=2))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(lstm2), axis=2))
    inner_prod = tf.reduce_sum(conv2*lstm2, axis=2)
    sim = inner_prod
    # sim = inner_prod / (norm1 * norm2 + 1e-3) # Fatal Error!
    sim = tf.nn.softmax(sim, dim=1)
    sim = tf.reshape(sim, [-1, hp.e_len, 1])
    att2_output = sim * conv2
    att2_output = tf.reduce_sum(att2_output, axis=1)
    # hello = tf.reduce_mean(lstm2+conv2, axis=1)
    linear = tf.layers.dense(inputs=att2_output, units=1, activation=tf.nn.sigmoid)

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

    test_essays, test_scores = test_set.all()

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
            train_essays, train_scores = train_set.next_batch(hp.batch_size)
            sess.run(train_op, feed_dict={essays: train_essays, scores: train_scores})
            gstep, lrate, summary, test_loss, pred_scores = sess.run([g_step, learning_rate, merged, loss, preds], feed_dict={essays:test_essays, scores: test_scores})

            qwk_value = qwk(pred_scores, test_scores, 1)
            print("iter: {} \t loss: {:.6f} \t learning rate: {:.6f} \t QWK: {:.6f}".format(gstep, test_loss, lrate, qwk_value))
            summary_writer.add_summary(summary, gstep)

            # if gstep % 10 == 0:
                # saver.save(sess, paths.model, global_step=gstep)

if __name__ == "__main__":
    main()
