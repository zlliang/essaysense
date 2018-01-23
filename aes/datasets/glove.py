"""TODO"""

import os
import codecs

import numpy as np

from ..configs import paths, hp

# path_glove = paths.glove_path
ner = ["@PERSON", "@ORGANIZATION", "@LOCATION", "@DATE", "@TIME", "@MONEY", "@PERCENT", "@MONTH", "@EMAIL", "@NUM", "@CAPS", "@DR", "@CITY", "@STATE"]

def load_glove(path_glove=paths.glove_path):
    """LOAD_GLOVE, TODO"""
    # print("Loading: GloVe word vectors")  # TODO
    try:
        with codecs.open(path_glove, 'r', 'UTF-8') as glove_file:
            glove_vectors = {}
            # add num and entity encoding
            for i in ner:
                glove_vectors[i] = np.random.randn(hp.w_dim)
            for item in glove_file.readlines():
                item_lst = item.strip().split(' ') # format the line to a list
                word = item_lst[0]
                vec = [float(i) for i in item_lst[1:]] # convert strings to floats
                glove_vectors[word] = np.array(vec)
        return glove_vectors
    except:
        raise NotImplementedError  # TODO
