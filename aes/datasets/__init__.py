"""ASAP DATASETS. TODO"""

from .asap import *
from .glove import *
from .preprocessing import *

# def lookup(sequence):
#     seq_mat = []
#     for i in sequence:
#         try:
#             seq_mat.append(lookup_table[i])
#         except KeyError:
#             seq_mat.append(np.random.randn(DIM))
#     return np.array(seq_mat)
