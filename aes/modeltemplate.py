import os
import tensorflow as tf

from aes.configs import paths
from aes.metrics import qwk

class Model:
    """Neural network models templeate on ASAP-AES dataset.

    This class describes general characteristics of our neural network
    models on ASAP-AES dataset. It's like an interface, defining
    standard behaviors of these models.

    Models shall be implemented into these methods:
        - define_graph: describes model graph.
        - train: trains the model.
    """
    def __init__(self, hyperparameters, train_set, test_set, domain_id):
        """Initialiazation of the neural network model.

        Args:
            - hyperparameters: an object that have properties representing
                               hyperparameters.
            - train_set: train dataset, "next_batch" method must be implemented.
            - test_set: test dataset, "all" method must be implemented.
            - domain_id: domain id of ASAP dataset, ranging 1-8.
        """
        self.hp = hyperparameters
        self.train_set = train_set
        self.test_set = test_set
        self.domain_id = str(domain_id)
        self.name = ""  # specific for subclasses

    def define_graph(self):
        """Define computational graph of the specific model.

        This is the interface of subclasses. Specific models must implement this
        method so that the computational process can be run appropriately.
        """
        raise NotImplementedError

    def train(self, save_path=None, summary_path=None):
        """Train the model using given train set.

        Args:
            - save_path: choose the model's checkpoint saving path. Default is
                         None, which means the train result would not be saved.
            - summary_path: choose the model's TensorBoard summary saving path.
                            Default is the path generated corresponding to your
                            system time.
        """
        print("[Training] " + self.name)
        if not save_path:
            save_path = paths.model(self.name)
        if not summary_path:
            summary_path = paths.summary(self.name)


        (essays,
         scores,
         merged_summary,
         loss,
         preds) = self.define_graph()
        g_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        learning_rate = tf.train.exponential_decay(self.hp.learning_rate,
                                                   global_step=g_step,
                                                   decay_steps=10,
                                                   decay_rate=0.93)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=g_step)

        (test_essays,
         test_scores) = self.test_set.all()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        for _ in range(self.hp.train_epochs):
            (train_essays,
             train_scores) = self.train_set.next_batch(self.hp.batch_size)
            sess.run(train_op,
                     feed_dict={essays: train_essays, scores: train_scores})
            (g_step_got,
             summary_got,
             loss_got,
             preds_got) = sess.run([g_step,
                                    merged_summary,
                                    loss,
                                    preds], feed_dict={essays: test_essays,
                                                       scores: test_scores})
            qwk_value = qwk(preds_got, test_scores, self.domain_id)
            print("Train: {:3d},   Loss: {:.6f},   QWK-on-dev-set: {:.6f}"
                  .format(g_step_got, loss_got, qwk_value))
            summary_writer.add_summary(summary_got, g_step_got)
            if g_step_got % 100 == 0:
                saver.save(sess, save_path)
        sess.close()

    def evaluate(self, saved_path=None):
        """Evaluate the model using given test set."""

        if not saved_path:
            saved_path = paths.model(self.name)
            saved_path_ckpt = paths.model_ckpt(self.name)

        ckpt = tf.train.get_checkpoint_state(saved_path_ckpt)
        if not ckpt or not ckpt.model_checkpoint_path:
            print("[Error] The model haven't been trained! Please train first!")
            return

        (essays,
         scores,
         _,
         _,
         preds) = self.define_graph()
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)
        (test_essays,
         test_scores) = self.test_set.all()
        preds_got = sess.run(preds, feed_dict={essays: test_essays,
                                               scores: test_scores})
        qwk_value = qwk(preds_got, test_scores, self.domain_id)
        print("[Evaluating {}] QWK-on-test-set: {:.6f}"
               .format(self.name, qwk_value))
        sess.close()

    def visualize(self, summary_path=None):
        """Visualize the model using TensorBoard."""

        print("[Visualizing] Calling Tensorboard...")
        if not summary_path:
            summary_path = paths.summary(self.name)

        if not os.path.exists(summary_path):
            print("[Error] The model haven't been trained! Please train first!")
            return

        os.system("tensorboard --logdir={}".format(summary_path))
