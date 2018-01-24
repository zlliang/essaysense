"""Datasets readers and utilities.

This submodule of AES project consists of automated datasets reader and
utilities. In this pre-release version, only ASAP-AES dataset is supported
to load as train or test set.
"""

# Dataset readers
from aes.datasets.asap import load_asap
from aes.datasets.glove import load_glove

# Classes for structuring ASAP-AES dataset
from aes.datasets.utils import SentenceLevelTrainSet
from aes.datasets.utils import SentenceLevelTestSet
from aes.datasets.utils import DocumentLevelTrainSet
from aes.datasets.utils import DocumentLevelTestSet
