"""Datasets readers and utilities.

This submodule of AES project consists of automated datasets reader and
utilities. In this pre-release version, only ASAP-AES dataset is supported
to load as train or test set.
"""

# Dataset readers
from essaysense.datasets.asap import load_asap
from essaysense.datasets.glove import load_glove

# Classes for structuring ASAP-AES dataset
from essaysense.datasets.utils import SentenceLevelTrainSet
from essaysense.datasets.utils import SentenceLevelTestSet
from essaysense.datasets.utils import DocumentLevelTrainSet
from essaysense.datasets.utils import DocumentLevelTestSet
