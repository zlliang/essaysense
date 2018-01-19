"""ASAP DATASETS. TODO"""

from .hyperparameters import hp
from .util import TrainSet
from .asap import load_asap
from .glove import load_glove

import numpy as np

train_set = TrainSet(load_asap, load_glove)
next_batch = train_set.next_batch
