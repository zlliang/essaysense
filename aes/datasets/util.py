import numpy as np
import nltk

from .hyperparameters import hp
from .asap import normalize_score

class ASAPDataSet:
    """TODO"""
    def __init__(self, lookup_table):
        """TODO"""
        self.s_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.lookup_table = lookup_table

    def tokenize(self, essay_text):
        """TODO"""
        essay_text = essay_text.lower()
        essay = []
        sentences = self.s_tokenizer.tokenize(essay_text)
        for sentence in sentences:
            essay.append(nltk.word_tokenize(sentence))
        return essay

    def lookup(self, word):
        try:
            return self.lookup_table[word]
        except KeyError:
            return np.zeros(hp.w_dim)

    def embed_essay(self, essay_text):
        """TODO!!! DOCSTRING NEEDED!"""
        essay_text = self.tokenize(essay_text)
        embedded = np.zeros([hp.e_len, hp.s_len, hp.w_dim])
        for i in range(min(len(essay_text), hp.e_len)):
            sentence = essay_text[i]
            for j in range(min(len(sentence), hp.s_len)):
                word = sentence[j]
                embedded[i, j] = self.lookup(word)
        return embedded

class TrainSet(ASAPDataSet):
    """TODO"""
    def __init__(self, load_train_set, lookup_table):
        """TODO"""
        super().__init__(lookup_table)
        self.train_set_generator = self.structure(load_train_set())

    def structure(self, raw_train_set):
        """TODO"""
        # TODO!!! DOCSTRING NEEDED! GENERATOR USED!
        set_size = len(raw_train_set)
        while True:
            i = np.random.randint(0, set_size)
            item = raw_train_set[i]
            yield {"domain_id":    item["essay_set"],
                   "essay": self.embed_essay(item["essay"]),
                   "score": normalize_score(item["essay_set"],
                                            item["domain1_score"])}

    def next_batch(self, size_demand):
        """TODO"""
        essays_batched = []
        scores_batched = []
        for _ in range(size_demand):
            next_item = next(self.train_set_generator)
            essays_batched.append(next_item["essay"])
            scores_batched.append(next_item["score"])
        essays_batched = np.array(essays_batched)
        scores_batched = np.array(scores_batched)
        return essays_batched, scores_batched

class TestSet(ASAPDataSet):
    """TODO"""
    def __init__(self, load_test_set, lookup_table):
        """TODO"""
        super().__init__(lookup_table)
        self._raw_test_set = load_test_set()

    def all(self):
        """TODO"""
        essays_all = []
        scores_all = []
        for item in self._raw_test_set:
            essays_all.append(self.embed_essay(item["essay"]))
            scores_all.append(normalize_score(item["essay_set"],
                                              item["domain1_score"]))
        essays_all = np.array(essays_all)
        scores_all = np.array(scores_all)
        return essays_all, scores_all
