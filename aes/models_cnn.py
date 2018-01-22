import tensorflow as tf

from .datasets import train_set, test_set
from .configs import paths, hp
from .qwk import qwk

def main():
    essays = tf.placeholder(tf.float32, [None, hp.e_len, hp.s_len, hp.w_dim])
    scores = tf.placeholder(tf.float32, [None])

    input_layer = tf.reshape(essays, [-1, hp.e_len, hp.s_len, hp.w_dim])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=hp.w_convunits_size,
        kernel_size=[1, hp.w_window_len],
        # kernel_initializer=tf.initializers.uniform_unit_scaling(),
        # bias_initializer=tf.initializers.uniform_unit_scaling(),
        padding="same",
        activation=None)

    bn1 = tf.layers.batch_normalization(inputs=conv1)

    activated1 = tf.nn.relu(bn1)

    pool1 = tf.layers.max_pooling2d(inputs=activated1, pool_size=[1, hp.s_len], strides=1)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=hp.s_convunits_size,
        kernel_size=[hp.s_window_len, 1],
        # kernel_initializer=tf.initializers.uniform_unit_scaling(),
        # bias_initializer=tf.initializers.uniform_unit_scaling(),
        padding="same",
        activation=None)

    bn2 = tf.layers.batch_normalization(inputs=conv2)

    activated2 = tf.nn.relu(bn2)

    pool2 = tf.layers.max_pooling2d(inputs=activated2, pool_size=[hp.e_len, 1], strides=1)

    pool2_flat = tf.reshape(pool2, [-1, hp.s_convunits_size])

    # dropout = tf.layers.dropout(inputs=pool2_flat, rate=0.4)

    dense1 = tf.layers.dense(inputs=pool2_flat, units=hp.hidden_size, activation=tf.nn.relu)

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
