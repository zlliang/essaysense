from aes.datasets import *
from functools import partial

domain1_train_set = TrainSet(partial(load_asap, path=path_train, domain_id="1"), glove_table)
domain2_train_set_sample = TrainSet(partial(load_asap, path=path_train, domain_id="2"), glove_table)

domain2_test_set = TestSet(partial(load_asap, path=path_train, domain_id="2"), glove_table)
