from os import path


class HyperParameters:
    """
    w for words
    s for sentences
    e for essay
    """

    def __init__(self):
        self._w_dim = 100
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
        self.aes_root = path.dirname(__file__)
        self.model = path.join(self.aes_root, "tfmodels", "models.ckpt")
        self.summary_train = path.join(self.aes_root, "tfsummaries", "train")
        self.summary_test = path.join(self.aes_root, "tfsummaries", "test")

        self.datasets_root = path.join(self.aes_root, "datasets")
        self.asap_path = path.join(self.datasets_root, "training_set_rel3.tsv")
        self.glove_path = path.join(self.datasets_root, "glove.6B.100d.txt")

hp = HyperParameters()
paths = ProjectPaths()
