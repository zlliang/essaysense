"""ASAP DATASETS. TODO"""

import os
from functools import partial

from ..configs import hp
from .util import TrainSet, TestSet, LSTM_TrainSet, LSTM_TestSet, MetaData
from .asap import load_asap
from .glove import load_glove

import numpy as np

path_asap = os.path.join(os.path.dirname(__file__), "training_set_rel3.tsv")
path_train = os.path.join(os.path.dirname(__file__), "train.tsv")
path_test =  os.path.join(os.path.dirname(__file__), "test.tsv")
load_train_1 = partial(load_asap, path=path_train, domain_id="1")
load_test_1 = partial(load_asap, path=path_test, domain_id="1")
glove_table = load_glove()


train_set = TrainSet(load_train_1, glove_table)
test_set = TestSet(load_test_1, glove_table)

lstm_train_set = LSTM_TrainSet(load_train_1, glove_table)
lstm_test_set = LSTM_TestSet(load_train_1, glove_table)
# meta = MetaData()
