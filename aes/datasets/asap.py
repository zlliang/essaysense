"""Automated Student Assessment Prize (ASAP) datasets reader.

This module get the ASAP data prepared.
"""

import os
import csv
import codecs
import urllib # TODO

path_asap = os.path.join(os.path.dirname(__file__), "training_set_rel3.tsv")
# url_asap =
set_valid = [str(i) for i in range(1, 9)]
score_valid = [str(i) for i in range(0, 61)]
def load_asap(path=path_asap, domain_id=None):
    """return asap_data list. TODO"""
    # print("Loading: ASAP dataset") # TODO
    try:
        with codecs.open(path, "r", "ISO-8859-2") as asap_file:
            asap_reader = csv.DictReader(asap_file, delimiter="\t")
            if not domain_id:
                asap_data = [item for item in asap_reader if item["essay"] and item["essay_set"] in set_valid and item["domain1_score"] in score_valid]
            if domain_id:
                asap_data = [item for item in asap_reader if item["essay"] and item["essay_set"] == str(domain_id) and item["domain1_score"] in score_valid]
        return asap_data
    except:
        raise NotImplementedError
        # print("Downloading: ASAP dataset") # TODO
        # urllib.request.urlretrieve()


score_range = {"1": (2, 12),
               "2": (1, 6),
               "3": (0, 3),
               "4": (0, 3),
               "5": (0, 4),
               "6": (0, 4),
               "7": (0, 30),
               "8": (0, 60)}

def normalize_score(domain_id, score):
    """TODO"""
    lo, hi = score_range[domain_id]
    score = float(score)
    return (score - lo) / (hi - lo)
