"""Automated Student Assessment Prize (ASAP) datasets reader.

This module get the ASAP data prepared.
"""

import os
import csv
import codecs
import urllib # TODO

path_asap = os.path.join(os.path.dirname(__file__), 'training_set_rel3.tsv')
# url_asap =

def load_asap():
    """return asap_data list. TODO"""
    # print("Loading: ASAP dataset") # TODO
    try:
        with codecs.open(path_asap, "r", "ISO-8859-2") as asap_file:
            asap_reader = csv.DictReader(asap_file, delimiter="\t")
            asap_data = list(asap_reader)
        return asap_data
    except:
        raise NotImplementedError
        # print("Downloading: ASAP dataset") # TODO
        # urllib.request.urlretrieve()
