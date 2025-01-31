import os
from shlex import join
import numpy as np


####################################################
# This tool is to collect neighbors, and reform them
# as numpy.array style for futher usage.
####################################################


GRAPHISH_NAME = "graphish_v2"
RATIOS = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

CURRENT_DIR = os.path.dirname(__file__)
DATASET_TYPES = [
    "anchor_href",
    "urls_in_html",
    "urls_in_har",
    "all_related",
]

if __name__ == "__main__":

    for dataset_type in DATASET_TYPES:

        for RATIO in RATIOS:

            # ------------ url_fqdn_neis ------------
            fqdn_url_relations = np.genfromtxt(
                os.path.join(
                    CURRENT_DIR,
                    f"../data/dataset/{GRAPHISH_NAME}/{dataset_type}/{dataset_type}_graphish_{RATIO}_part/fu.txt",
                )
            )

            url_fqdn_neis = {}
            for fqdn_url_relation in fqdn_url_relations:
                fqdn = fqdn_url_relation[0]
                url = fqdn_url_relation[1]
                if url not in url_fqdn_neis:
                    url_fqdn_neis[url] = []
                url_fqdn_neis[url].append(fqdn)

            fqdn_urls = sorted(url_fqdn_neis.keys())

            url_fqdn_neis = [url_fqdn_neis[url] for url in fqdn_urls]
            url_fqdn_neis = np.array(
                [np.array(neis) for neis in url_fqdn_neis], dtype=object
            )

            np.save(
                os.path.join(
                    CURRENT_DIR,
                    f"../data/dataset/{GRAPHISH_NAME}/{dataset_type}/{dataset_type}_graphish_{RATIO}_part/nei_f.npy",
                ),
                url_fqdn_neis,
            )

            fqnd_url_len = [len(i) for i in url_fqdn_neis]
            print(max(fqnd_url_len), min(fqnd_url_len), np.mean(fqnd_url_len))

            # ------------ fqdn_registered_domain_neis ------------
            fqdn_registered_domain_relations = np.genfromtxt(
                os.path.join(
                    CURRENT_DIR,
                    f"../data/dataset/{GRAPHISH_NAME}/{dataset_type}/{dataset_type}_graphish_{RATIO}_part/fr.txt",
                )
            )

            fqdn_registered_domain_neis = {}
            for fqdn_registered_domain_relation in fqdn_registered_domain_relations:
                fqdn = fqdn_registered_domain_relation[0]
                registered_domain = fqdn_registered_domain_relation[1]
                if fqdn not in fqdn_registered_domain_neis:
                    fqdn_registered_domain_neis[fqdn] = []
                fqdn_registered_domain_neis[fqdn].append(registered_domain)

            fqdn_registered_domains = sorted(fqdn_registered_domain_neis.keys())

            fqdn_registered_domain_neis = [fqdn_registered_domain_neis[fqdn] for fqdn in fqdn_registered_domains]
            fqdn_registered_domain_neis = np.array(
                [np.array(neis) for neis in fqdn_registered_domain_neis]
            )

            np.save(
                os.path.join(
                    CURRENT_DIR,
                    f"../data/dataset/{GRAPHISH_NAME}/{dataset_type}/{dataset_type}_graphish_{RATIO}_part/nei_r.npy",
                ),
                fqdn_registered_domain_neis,
            )

            fqdn_registered_domain_len = [len(i) for i in fqdn_registered_domain_neis]
            print(max(fqdn_registered_domain_len), min(fqdn_registered_domain_len), np.mean(fqdn_registered_domain_len))

            # ------------ url_word_neis ------------
            word_url_relations = np.genfromtxt(
                os.path.join(
                    CURRENT_DIR,
                    f"../data/dataset/{GRAPHISH_NAME}/{dataset_type}/{dataset_type}_graphish_{RATIO}_part/wu.txt",
                )
            )

            url_word_neis = {}
            for word_url_relation in word_url_relations:
                word = word_url_relation[0]
                url = word_url_relation[1]
                if url not in url_word_neis:
                    url_word_neis[url] = []
                url_word_neis[url].append(word)

            word_urls = sorted(url_word_neis.keys())

            url_word_neis = [url_word_neis[url] for url in word_urls]
            url_word_neis = np.array(
                [np.array(neis) for neis in url_word_neis], dtype=object
            )

            np.save(
                os.path.join(
                    CURRENT_DIR,
                    f"../data/dataset/{GRAPHISH_NAME}/{dataset_type}/{dataset_type}_graphish_{RATIO}_part/nei_w.npy",
                ),
                url_word_neis,
            )

            # give some basic statistics about neighbors
            word_url_len = [len(i) for i in url_word_neis]
            print(max(word_url_len), min(word_url_len), np.mean(word_url_len))
