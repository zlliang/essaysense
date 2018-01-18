"""TODO"""

import os
import codecs

from functools import partial

import numpy as np

path_glove = os.path.join(os.path.dirname(__file__), 'glove.6B.50d.txt')

def load_glove():
    """LOAD_GLOVE, TODO"""
    print("Loading: GloVe word vectors")
    try:
        with codecs.open(DIR_GLOVE, 'r', 'UTF-8') as glove_file:
            glove_vectors = {}
            for item in glove_file.readlines():
                item_lst = item.strip().split(' ') # format the line to a list
                word = item_lst[0]
                vec = [float(i) for i in item_lst[1:]] # convert strings to floats
                glove_vectors[word] = np.array(vec)
        return glove_vectors
    except:
        raise NotImplementedError  # TODO
