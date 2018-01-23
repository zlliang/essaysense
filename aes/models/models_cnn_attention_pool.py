import tensorflow as tf

from aes.datasets import train_set, test_set
from aes.configs import paths, hp
from aes.qwk import qwk

def main():
    essays = tf.placeholder(tf.float32, [None, hp.e_len, hp.s_len, hp.w_dim])
    scores = tf.placeholder(tf.float32, [None])

    input_layer = tf.reshape(essays, [-1, hp.e_len, hp.s_len, hp.w_dim])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=hp.w_convunits_size,
        kernel_size=[1, hp.w_window_len],
        padding="same",
        activation=tf.nn.relu)

    # attention pooling 1 TODO
    att1_mat = tf.Variable(tf.truncated_normal([hp.w_convunits_size, hp.w_convunits_size]), dtype=tf.float32)
    att1_bias = tf.Variable(tf.truncated_normal([1, 1, 1, hp.w_convunits_size]), dtype=tf.float32)
    att1_weight = tf.tensordot(conv1, att1_mat, axes=[3, 0]) + att1_bias
    att1_weight = tf.nn.tanh(att1_weight)
    att1_vec = tf.Variable(tf.truncated_normal([hp.w_convunits_size, 1]), dtype=tf.float32)
    att1_weight = tf.tensordot(att1_weight, att1_vec, axes=[3, 0])
        # attempt 1 TODO
    # att1_weight = tf.reduce_mean(att1_weight, axis=0)
    # att1_weight = tf.reduce_mean(att1_weight, axis=0)
    # att1_weight = tf.nn.softmax(att1_weight, dim=0)
    # att1_output = tf.tensordot(conv1, att1_weight, axes=[2, 0])
    # att1_output = tf.reshape(att1_output, [-1, hp.e_len, 1, hp.w_convunits_size])
        # attempt 2 TODO Seems right
    att1_weight = tf.nn.softmax(att1_weight, dim=2)
    att1_output = att1_weight * conv1
    att1_output = tf.reduce_sum(att1_output, axis=2, keep_dims=True)


    conv2 = tf.layers.conv2d(
        inputs=att1_output,
        filters=hp.s_convunits_size,
        kernel_size=[hp.s_window_len, 1],
        padding="same",
        activation=tf.nn.relu)

    att2_mat = tf.Variable(tf.truncated_normal([hp.s_convunits_size, hp.s_convunits_size]), dtype=tf.float32)
    att2_bias = tf.Variable(tf.truncated_normal([1, 1, 1, hp.s_convunits_size]), dtype=tf.float32)
    att2_weight = tf.tensordot(conv2, att2_mat, axes=[3, 0]) + att2_bias
    att2_weight = tf.nn.tanh(att2_weight)
    att2_vec = tf.Variable(tf.truncated_normal([hp.s_convunits_size, 1]), dtype=tf.float32)
    att2_weight = tf.tensordot(att2_weight, att2_vec, axes=[3, 0])
        # attempt 1 TODO
    # att2_weight = tf.reduce_mean(att2_weight, axis=0) # TODO
    # att2_weight = tf.nn.softmax(att2_weight, dim=0)
    # att2_output = tf.tensordot(conv2, att2_weight, axes=[1, 0])
    # att2_output = tf.reshape(att2_output, [-1, hp.s_convunits_size])
        # attempt 2 TODO seems right
    att2_weight = tf.nn.softmax(att2_weight, dim=1)
    att2_output = att2_weight * conv2
    att2_output = tf.reduce_sum(att2_output, axis=1, keep_dims=True)

    dense1 = tf.layers.dense(inputs=att2_output, units=hp.hidden_size, activation=tf.nn.relu)

    dense2 = tf.layers.dense(inputs=dense1, units=1, activation=tf.nn.sigmoid)

    preds = tf.reshape(dense2, [-1])
    tf.summary.histogram('preds', preds)


    loss = tf.losses.mean_squared_error(scores, preds)
    tf.summary.scalar('loss', loss)


    g_step = tf.Variable(0, dtype=tf.int64, trainable=False)

    learning_rate = tf.train.exponential_decay(hp.learning_rate, global_step=g_step, decay_steps=10, decay_rate=0.93)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss)

    add_global = g_step.assign_add(1)

    test_essays, test_scores = test_set.all()

    # human_scores_tensor = tf.constant(test_scores, tf.float32)
    # tf.summary.histogram('human_scores', human_scores_tensor)
    merged = tf.summary.merge_all()



    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(paths.summary_train, sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            train_essays, train_scores = train_set.next_batch(hp.batch_size)
            sess.run([train_op, add_global], feed_dict={essays: train_essays, scores: train_scores})
            gstep, lrate, summary, test_loss, pred_scores = sess.run([g_step, learning_rate, merged, loss, preds], feed_dict={essays:test_essays, scores: test_scores})

            qwk_value = qwk(pred_scores, test_scores, 1)
            print("iter: {} \t loss: {:.6f} \t learning rate: {:.6f} \t QWK: {:.6f}".format(gstep, test_loss, lrate, qwk_value))
            summary_writer.add_summary(summary, i)

if __name__ == "__main__":
    main()
