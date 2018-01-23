from os import path
import time


class HyperParameters:
    """
    w for words
    s for sentences
    e for essay
    """

    def __init__(self):
        self._w_dim = 50
        self._s_len = 20
        self._e_len = 60
        self._w_window_len = 5
        self._s_window_len = 3
        self._w_convunits_size = 64
        self._s_convunits_size = 32
        self._hidden_size = 100
        self._batch_size = 20
        self._learning_rate = 0.006
        self._stddev = 0.1
        self._dropout_keep_prob = 0.5

        # NEW 1-21 for LSTM model
        self.lstm_e_len = 500  # word-count for whole essay
        self.lstm_convunits_size = 64
        self.lstm_hidden_size = 200
        self.lstm_dropout_keep_prob = 0.3
        self.lstm_sen_level_convunits_size = 80
        self.lstm_sen_level_att_pool_hidden_size = 50

        self.max_grad_norm = 5

    @property
    def w_dim(self):
        return self._w_dim

    @property
    def s_len(self):
        return self._s_len

    @property
    def e_len(self):
        return self._e_len

    @property
    def w_window_len(self):
        return self._w_window_len

    @property
    def s_window_len(self):
        return self._s_window_len

    @property
    def w_convunits_size(self):
        return self._w_convunits_size

    @property
    def s_convunits_size(self):
        return self._s_convunits_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def stddev(self):
        return self._stddev

    @property
    def dropout_keep_prob(self):
        return self._dropout_keep_prob

class ProjectPaths:
    def __init__(self):
        self.now = time.strftime("%m-%d-%H.%M", time.localtime())
        self.aes_root = "aes"
        self.model_ckpt = path.join(self.aes_root, "tfmetadata")

        self.model = path.join(self.aes_root, "tfmetadata", "models.ckpt")
        self.summary_train = path.join(self.aes_root, "tfmetadata", "summary_train_"+self.now)
        self.summary_test = path.join(self.aes_root, "tfmetadata", "summary_test_"+self.now)

        self.model_test = path.join(self.summary_test, "models.ckpt")

        self.datasets_root = path.join(self.aes_root, "datasets")
        self.asap_path = path.join(self.datasets_root, "training_set_rel3.tsv")
        self.glove_path = path.join(self.datasets_root, "glove.6B.50d.txt")

hyperparameters = HyperParameters()
hp = hyperparameters
paths = ProjectPaths()
