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
        self._s_windows_len = 3
        self._w_convunits_size = 50
        self._s_convunits_size = 50
        self._hidden_size = 50
        self._batch_size = 5
        self._learning_rate = 0.01

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

hp = HyperParameters()
