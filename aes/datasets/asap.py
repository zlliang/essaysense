import os
import csv
import codecs
import urllib.request

from aes.configs import paths

# Some metadata about ASAP-AES dataset, in order to extract valid items.
set_valid = [str(i) for i in range(1, 9)]
score_valid = [str(i) for i in range(0, 61)]

def load_asap(path=paths.asap, domain_id=None):
    """Read ASAP-AES dataset from Tab Separated Value (TSV) file.

    Note that this function would automatically detect or download ASAP-AES
    dataset.

    Args:
        - path (option): identifies ASAP data file manually.
        - domain_id (option) : identifies the specific prompt of data,
                               everything if not given.

    Returns:
        - Raw ASAP-AES data of OrderedDict, in which "essays", "essay_set" and
          "domain1_score" is crucial.
    """
    if domain_id:
        print("[Loading] ASAP-AES domain {} dataset...".format(domain_id))
    else:
        print("[Loading] ASAP-AES dataset...")
    try:
        with codecs.open(path, "r", "ISO-8859-2") as asap_file:
            asap_reader = csv.DictReader(asap_file, delimiter="\t")
            # Extract valid items in the dataset.
            if not domain_id:
                asap_data = [item for item in asap_reader
                             if item["essay"]
                             and item["essay_set"] in set_valid
                             and item["domain1_score"] in score_valid]
            else:
                asap_data = [item for item in asap_reader
                             if item["essay"]
                             and item["essay_set"] == str(domain_id)
                             and item["domain1_score"] in score_valid]
        return asap_data
    except FileNotFoundError:
        # Auto download.
        if path == paths.asap:
            print("[Downlading] ASAP-AES dataset...")
            urllib.request.urlretrieve(paths.asap_url, path)
            return load_asap(path, domain_id)
        else:
            print("[Error] Seems you identified an non-existing dataset...")
