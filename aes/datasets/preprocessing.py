from .asap import load_asap
from .glove import load_glove
import numpy as np

import nltk

lookup_table = load_glove()
asap = load_asap()

def lookup(word):
    try:
        return lookup_table[word]
    except KeyError:
        return np.zeros(50)

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def tokenize(essay_text):
    essay_text = essay_text.lower()  # LOWER!!! DOCSTRING NEEDED!!!!
    essay = []
    sentences = sentence_tokenizer.tokenize(essay_text)
    for sentence in sentences:
        essay.append(nltk.word_tokenize(sentence))
    return essay

def essay_embedding(essay, w_dim, s_len, e_len):
    """TODO!!! DOCSTRING NEEDED!"""
    embedded = np.zeros([e_len, s_len, w_dim])
    for i in range(min(len(essay), e_len)):
        sentence = essay[i]
        for j in range(min(len(sentence), s_len)):
            word = sentence[j]
            embedded[i, j] = lookup(word)
    return embedded




# essay_set = []
# for item in asap:
#     if item["essay"]:
#         essay_set.append({"id":    item["essay_id"],
#                           "essay": item["essay"],
#                           "score": item["domain1_score"]})
