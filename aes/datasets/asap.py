"""Automated Student Assessment Prize (ASAP) datasets reader.

This module get the ASAP data prepared.
"""

import os
import csv
import codecs
from urllib import request

from ..configs import paths

path_asap = paths.asap_path
url_asap = "http://p2u3jfd2o.bkt.clouddn.com/training_set_rel3.tsv"

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
        request.urlretrieve(url_asap, path_asap)
        load_asap()
