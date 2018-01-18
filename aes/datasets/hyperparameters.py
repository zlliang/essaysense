class HyperParameters:
    """
    w for words
    s for sentences
    e for essay
    """
    def __init__(self):
        pass

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
    def learning_rate(self):
        return self._learning_rate
