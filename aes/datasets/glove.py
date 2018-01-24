import os
import codecs
import urllib.request
import numpy as np

from aes.configs import paths
from aes.configs import hyperparameters as hp

# Metadata of Named Entities Recogonition (NER) and other replacements.
ner = ["@PERSON", "@ORGANIZATION", "@LOCATION", "@DATE",
       "@TIME", "@MONEY", "@PERCENT", "@MONTH", "@EMAIL",
       "@NUM", "@CAPS", "@DR", "@CITY", "@STATE"]

def load_glove(path=paths.glove):
    """Read GloVe word embedding data from plain text.

    Note that this function would automatically detect or download GloVe
    dataset.

    Args:
        - path (option): identifies GloVe data file manually.

    Returns:
        - Raw GloVe data of dict, every entry of which is a NumPy vector.
    """
    print("[Loading] GloVe word vectors...")
    try:
        with codecs.open(path, 'r', 'UTF-8') as glove_file:
            glove_vectors = {}
            # Add numbers and NER embedding entries.
            for i in ner:
                glove_vectors[i] = np.random.randn(hp.w_dim)
            for item in glove_file.readlines():
                item_lst = item.strip().split(' ')
                word = item_lst[0]
                vec = [float(i) for i in item_lst[1:hp.w_dim+2]]
                glove_vectors[word] = np.array(vec)
        return glove_vectors
    except FileNotFoundError:
        # Auto download.
        print("[Downlading] GloVe word embedding dataset...")
        urllib.request.urlretrieve(paths.glove_url, path)
        return load_glove(path)
