import re
import numpy as np
import nltk

class ASAPDataSet:
    """Class for ASAP-AES dataset preprocessors.

    This class contains several automated preprocessing utilities for ASAP-AES
    dataset, including score ranges metadata, tools for mormalizing scores to
    [0, 1] interval, tools for tokenizing and embedding essay texts to word
    embedding matrices, etc.

    Note that in our experiments, both sentence-level and document-level models
    are implemented. The former means we should tokenize essay texts into
    sentences, and tokenize sentences into words. The latter one means to toke-
    nize essay texts just into word sequences.
    """
    def __init__(self, hyperparameters, lookup_table):
        """Constructor for initializing ASAP-AES datasets.

        Args:
            - hyperparameters TODO: hyperparameters of the experiments.
            - lookup_table: word embedding lookup table, which should be a dict
                            mapping words into their NumPy vector repre-
                            sentation.
        """
        # This constructor tries to detect or download NLTK's tokenizer
        # automatically.
        try:
            self.s_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        except LookupError:
            nltk.download("punkt")
            self.s_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # Also load hyperparameters and lookup table.
        self.lookup_table = lookup_table
        self.hp = hyperparameters

    @property
    def score_range(self):
        """Returns score ranges metadata of ASAP-AES."""
        return {"1": (2, 12),
                "2": (1, 6),
                "3": (0, 3),
                "4": (0, 3),
                "5": (0, 4),
                "6": (0, 4),
                "7": (0, 30),
                "8": (0, 60)}

    def normalize_score(self, domain_id, score):
        """Normalize scores into [0, 1] interval.

        Args:
            - domain_id: prompt id of ASAP dataset, ranging 1-8.
            - score: the score to be normalized. Note that the score should cor-
                     responding to the appropriate domain id.

        Returns:
            - normalized score of float number type.
        """
        lo, hi = self.score_range[str(domain_id)]
        score = float(score)
        return (score - lo) / (hi - lo)

    def lookup(self, word):
        """Lookup a word embedding.

        Args:
            - word: the word to be lookup.

        Returns:
            - word embedding vector in the form of NumPy ndarray.
        """
        try:
            return self.lookup_table[word]
        except KeyError:
            # If a unknown word appers, treat it as a zero vector.
            return np.zeros(self.hp.w_dim)

    def tokenize(self, essay_text):
        """Deprecated TODO"""
        essay_text = essay_text.lower()
        essay = []
        sentences = self.s_tokenizer.tokenize(essay_text)
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            for index, token in enumerate(tokens):
                if token == "@" and (index+1) < len(tokens):
                    tokens[index+1] = "@" + re.sub("[0-9]+.*", "", tokens[index+1].upper())
                    tokens.pop(index)
            essay.append(tokens)
        return essay

    def sentence_level_tokenize(self, essay_text):
        """Tokenize essay text on sentence level.

        This method tokenize essay text into nesting lists, consisting of
        sentences lists in which words are tokenized. In other words, an essay
        would be tokenized like:

            "Good boy! And Good girl!"
            -> [["Good", "boy", "!"], ["And", "Good", "Girl", "!"]]

        Note that named entities recoginition (NER) and some replacements like
        "@PERSON1" and "@ORGANIZATION2" would be detected and treat as a single
        word if they are of the same replacement type.

        Args:
            - essay_text: essay text of string type.

        Returns:
            - nested list consisting of tokenized essay, like the format above.
        """
        essay_text = essay_text.lower()  # Use lower-cases for word embeddings.
        essay = []
        sentences = self.s_tokenizer.tokenize(essay_text)
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            # Detect and process NER and replacements following.
            for index, token in enumerate(tokens):
                if token == "@" and (index+1) < len(tokens):
                    tokens[index+1] = "@" + \
                        re.sub("[0-9]+.*", "", tokens[index+1].upper())
                    tokens.pop(index)
            essay.append(tokens)
        return essay

    def embed_essay(self, essay_text):
        """Deprecated TODO"""
        essay_text = self.tokenize(essay_text)
        embedded = np.zeros([self.hp.e_len, self.hp.s_len, self.hp.w_dim])
        for i in range(min(len(essay_text), self.hp.e_len)):
            sentence = essay_text[i]
            for j in range(min(len(sentence), self.hp.s_len)):
                word = sentence[j]
                embedded[i, j] = self.lookup(word)
        return embedded

    def sentence_level_embed(self, essay_text):
        """Complete word embedding for an essay text on sentence level.

        Args:
            - essay_text: essay text of string type.

        Returns:
            - embedded essay NumPy tensor in the shape of:
              (sentence number, word number, word embedding dimension).
              These numbers are defined in the hyperparameters.
        """
        essay_text = self.sentence_level_tokenize(essay_text)
        embedded = np.zeros([self.hp.e_len, self.hp.s_len, self.hp.w_dim])
        for i in range(min(len(essay_text), self.hp.e_len)):
            sentence = essay_text[i]
            for j in range(min(len(sentence), self.hp.s_len)):
                word = sentence[j]
                embedded[i, j] = self.lookup(word)
        return embedded

    def document_level_tokenize(self, essay_text):
        """Tokenize essay text on document level.

        This method tokenize essay text into a word sequence, like this:

            "Good boy! And Good girl!"
            -> ["Good", "boy", "!", "And", "Good", "Girl", "!"]

        Note that named entities recoginition (NER) and some replacements like
        "@PERSON1" and "@ORGANIZATION2" would be detected and treat as a single
        word if they are of the same replacement type.

        Args:
            - essay_text: essay text of string type.

        Returns:
            - A list consisting of tokenized essay, like the format above.
        """
        essay_text = essay_text.lower()  # Use lower-cases for word embeddings.
        essay = nltk.word_tokenize(essay_text)
        # Detect and process NER and replacements following.
        for index, token in enumerate(essay):
            if token == "@" and (index+1) < len(essay):
                essay[index+1] = "@" + \
                    re.sub("[0-9]+.*", "", essay[index+1].upper())
                essay.pop(index)
        return essay

    def document_level_embed(self, essay_text):
        """Complete word embedding for an essay text on document level.

        Args:
            - essay_text: essay text of string type.

        Returns:
            - embedded essay NumPy tensor in the shape of:
              (word number, word embedding dimension).
              These numbers are defined in the hyperparameters.
        """
        essay_text = self.document_level_tokenize(essay_text)
        embedded = np.zeros([self.hp.d_e_len, self.hp.w_dim])
        for i in range(min(len(essay_text), self.hp.d_e_len)):
            embedded[i] = self.lookup(essay_text[i])
        return embedded

    def lstm_embed_essay(self, essay_text):
        """Deprecated TODO"""
        essay_text = essay_text.lower()
        essay_text = nltk.word_tokenize(essay_text)
        for index, token in enumerate(essay_text):
            if token == "@" and (index+1) < len(essay_text):
                essay_text[index+1] = "@" + re.sub("[0-9]+.*", "", essay_text[index+1].upper())
                essay_text.pop(index)
        embedded = np.zeros([self.hp.d_e_len, self.hp.w_dim])
        for i in range(min(len(essay_text), self.hp.d_e_len)):
            embedded[i] = self.lookup(essay_text[i])
        return embedded

class SentenceLevelTrainSet(ASAPDataSet):
    """Class for sentence-level ASAP-AES train sets.

    Instantiation of this class will create a structured train set including
    a generator that could generate essays and scores batch.
    """
    def __init__(self, hyperparameters, lookup_table, raw_train_set):
        """Constructor for initializing sentence-level ASAP-AES train sets.

        Args:
            - hyperparameters TODO: hyperparameters of the experiments.
            - lookup_table: word embedding lookup table, which should be a dict
                            mapping words into their NumPy vector repre-
                            sentation.
            - raw_train_set: OrderedDict generated by ASAP TSV file reader,
                             see "datasets/asap.py".
        """
        super().__init__(hyperparameters, lookup_table)
        self.train_set_generator = self.structure(raw_train_set)

    def structure(self, raw_train_set):
        """Structure train set into needed format.

        Note that this method uses Python's generator feature to realize
        lazy-loading, in order to promote running speed.

        Args:
            - raw_train_set: OrderedDict generated by ASAP TSV file reader,
                             see "datasets/asap.py".
        Returns:
            - A generator that generates a structured ramdom essay item dict in
              the following format:
              \{
                "domain_id": domain id of the essay text,
                "essay": sentence-level embedded essay,
                "score": gold score of the essay
              \}
        """
        set_size = len(raw_train_set)
        while True:
            i = np.random.randint(0, set_size)
            item = raw_train_set[i]
            yield {"domain_id": item["essay_set"],
                   "essay": self.sentence_level_embed(item["essay"]),
                   "score": self.normalize_score(item["essay_set"],
                                                 item["domain1_score"])}

    def next_batch(self, size_demand):
        """Generate a batch of sentence-level train data.

        This methods generates a tuple of train data (essays, scores) to feed
        our neural network models.

        Args:
            - size_demand: demand batch size of integer.
        Returns (tupled):
            - essay_batched: NumPy tensor of the following shape
                             (batch size, essay len, sentence len, word dim),
                             of which numbers are identified by hyperparameters.
            - scores_batched: 1D NumPy array of batch scores.
        """
        essays_batched = []
        scores_batched = []
        for _ in range(size_demand):
            next_item = next(self.train_set_generator)  # Generate next item
            essays_batched.append(next_item["essay"])
            scores_batched.append(next_item["score"])
        essays_batched = np.array(essays_batched)
        scores_batched = np.array(scores_batched)
        return essays_batched, scores_batched

class SentenceLevelTestSet(ASAPDataSet):
    """Class for sentence-level ASAP-AES test sets.

    Instantiation of this class will create a structured test set including
    a method generating all structured data ready to feed NN models.
    """
    def __init__(self, hyperparameters, lookup_table, raw_test_set):
        """Constructor for initializing sentence-level ASAP-AES train sets.

        Args:
            - hyperparameters TODO: hyperparameters of the experiments.
            - lookup_table: word embedding lookup table, which should be a dict
                            mapping words into their NumPy vector repre-
                            sentation.
            - raw_train_set: OrderedDict generated by ASAP TSV file reader,
                             see "datasets/asap.py".
        """
        super().__init__(hyperparameters, lookup_table)
        self._raw_test_set = raw_test_set

    def all(self):
        """Generate a batch of sentence-level test data.

        This methods generates a tuple of test data (essays, scores) including
        all test set to feed our neural network models.

        Returns (tupled):
            - essay_all: NumPy tensor of the following shape
                         (test set size, essay len, sentence len, word dim),
                         of which numbers are identified by hyperparameters.
            - scores_all: 1D NumPy array of all scores.
        """
        essays_all = []
        scores_all = []
        for item in self._raw_test_set:
            essays_all.append(self.sentence_level_embed(item["essay"]))
            scores_all.append(self.normalize_score(item["essay_set"],
                                                   item["domain1_score"]))
        essays_all = np.array(essays_all)
        scores_all = np.array(scores_all)
        return essays_all, scores_all

class DocumentLevelTrainSet(ASAPDataSet):
    """Class for document-level ASAP-AES train sets.

    Instantiation of this class will create a structured train set including
    a generator that could generate essays and scores batch.
    """
    def __init__(self, hyperparameters, lookup_table, raw_train_set):
        """Constructor for initializing document-level ASAP-AES train sets.

        Args:
            - hyperparameters TODO: hyperparameters of the experiments.
            - lookup_table: word embedding lookup table, which should be a dict
                            mapping words into their NumPy vector repre-
                            sentation.
            - raw_train_set: OrderedDict generated by ASAP TSV file reader,
                             see "datasets/asap.py".
        """
        super().__init__(hyperparameters, lookup_table)
        self.train_set_generator = self.structure(raw_train_set)

    def structure(self, raw_train_set):
        """Structure train set into needed format.

        Note that this method uses Python's generator feature to realize
        lazy-loading, in order to promote running speed.

        Args:
            - raw_train_set: OrderedDict generated by ASAP TSV file reader,
                             see "datasets/asap.py".
        Returns:
            - A generator that generates a structured ramdom essay item dict in
              the following format:
              \{
                "domain_id": domain id of the essay text,
                "essay": sentence-level embedded essay,
                "score": gold score of the essay
              \}
        """
        set_size = len(raw_train_set)
        while True:
            i = np.random.randint(0, set_size)
            item = raw_train_set[i]
            yield {"domain_id": item["essay_set"],
                   "essay": self.document_level_embed(item["essay"]),
                   "score": self.normalize_score(item["essay_set"],
                                                 item["domain1_score"])}

    def next_batch(self, size_demand):
        """Generate a batch of document-level train data.

        This methods generates a tuple of train data (essays, scores) to feed
        our neural network models.

        Args:
            - size_demand: demand batch size of integer.
        Returns (tupled):
            - essay_batched: NumPy tensor of the following shape
                             (batch size, essay len, word dim),
                             of which numbers are identified by hyperparameters.
            - scores_batched: 1D NumPy array of batch scores.
        """
        essays_batched = []
        scores_batched = []
        for _ in range(size_demand):
            next_item = next(self.train_set_generator)  # Generate next item
            essays_batched.append(next_item["essay"])
            scores_batched.append(next_item["score"])
        essays_batched = np.array(essays_batched)
        scores_batched = np.array(scores_batched)
        return essays_batched, scores_batched

class DocumentLevelTestSet(ASAPDataSet):
    """Class for document-level ASAP-AES test sets.

    Instantiation of this class will create a structured test set including
    a method generating all structured data ready to feed NN models.
    """
    def __init__(self, hyperparameters, lookup_table, raw_test_set):
        """Constructor for initializing document-level ASAP-AES train sets.

        Args:
            - hyperparameters TODO: hyperparameters of the experiments.
            - lookup_table: word embedding lookup table, which should be a dict
                            mapping words into their NumPy vector repre-
                            sentation.
            - raw_train_set: OrderedDict generated by ASAP TSV file reader,
                             see "datasets/asap.py".
        """
        super().__init__(hyperparameters, lookup_table)
        self._raw_test_set = raw_test_set

    def all(self):
        """Generate a batch of document-level test data.

        This methods generates a tuple of test data (essays, scores) including
        all test set to feed our neural network models.

        Returns (tupled):
            - essay_all: NumPy tensor of the following shape
                         (test set size, essay len, word dim),
                         of which numbers are identified by hyperparameters.
            - scores_all: 1D NumPy array of all scores.
        """
        essays_all = []
        scores_all = []
        for item in self._raw_test_set:
            essays_all.append(self.document_level_embed(item["essay"]))
            scores_all.append(self.normalize_score(item["essay_set"],
                                                   item["domain1_score"]))
        essays_all = np.array(essays_all)
        scores_all = np.array(scores_all)
        return essays_all, scores_all



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
                   "score": self.normalize_score(item["essay_set"],
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
            scores_all.append(self.normalize_score(item["essay_set"],
                                                   item["domain1_score"]))
        essays_all = np.array(essays_all)
        scores_all = np.array(scores_all)
        return essays_all, scores_all


class LSTM_TrainSet(ASAPDataSet):
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
                   "essay": self.lstm_embed_essay(item["essay"]),
                   "score": self.normalize_score(item["essay_set"],
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

class LSTM_TestSet(ASAPDataSet):
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
            essays_all.append(self.lstm_embed_essay(item["essay"]))
            scores_all.append(self.normalize_score(item["essay_set"],
                                                   item["domain1_score"]))
        essays_all = np.array(essays_all)
        scores_all = np.array(scores_all)
        return essays_all, scores_all


class CrossDomain_TrainSet(ASAPDataSet):
    """TODO"""
    def __init__(self, load_train_set, lookup_table, sample_value):
        """TODO"""
        super().__init__(lookup_table)
        self.sample_value = sample_value
        self.essays_all = []
        self.scores_all = []
        for item in load_train_set()[:self.sample_value]:
            self.essays_all.append(self.embed_essay(item["essay"]))
            self.scores_all.append(self.normalize_score(item["essay_set"],
                                                   item["domain1_score"]))

    def next_batch(self, size_demand):
        """TODO"""
        essays_batched = []
        scores_batched = []
        for i in range(size_demand):
            randindex = np.random.randint(0, self.sample_value)
            essays_batched.append(self.essays_all[randindex])
            scores_batched.append(self.scores_all[randindex])
        essays_batched = np.array(essays_batched)
        scores_batched = np.array(scores_batched)
        return essays_batched, scores_batched

class CrossDomain_LSTM_TrainSet(ASAPDataSet):
    """TODO"""
    def __init__(self, load_train_set, lookup_table, sample_value):
        """TODO"""
        super().__init__(lookup_table)
        self.sample_value = sample_value
        self.essays_all = []
        self.scores_all = []
        for item in load_train_set()[:self.sample_value]:
            self.essays_all.append(self.lstm_embed_essay(item["essay"]))
            self.scores_all.append(self.normalize_score(item["essay_set"],
                                                   item["domain1_score"]))

    def next_batch(self, size_demand):
        """TODO"""
        essays_batched = []
        scores_batched = []
        for i in range(size_demand):
            randindex = np.random.randint(0, self.sample_value)
            essays_batched.append(self.essays_all[randindex])
            scores_batched.append(self.scores_all[randindex])
        essays_batched = np.array(essays_batched)
        scores_batched = np.array(scores_batched)
        return essays_batched, scores_batched
