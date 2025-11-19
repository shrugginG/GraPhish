import os
import numpy as np

DATASET_ROOT = "dataset"
NUM_PARTS = 5

CURRENT_DIR = os.path.dirname(__file__)
GRAPHISH_BASE = os.path.join(CURRENT_DIR, f"../data/{DATASET_ROOT}/graphish")

if __name__ == "__main__":

    for part_num in range(NUM_PARTS):
        part_dir = os.path.join(GRAPHISH_BASE, f"graphish_part{part_num}")

        # ------------ url_fqdn_neis ------------
        fqdn_url_relations = np.genfromtxt(os.path.join(part_dir, "fu.txt"))

        url_fqdn_neis = {}
        for fqdn_url_relation in fqdn_url_relations:
            fqdn = fqdn_url_relation[0]
            url = fqdn_url_relation[1]
            url_fqdn_neis.setdefault(url, []).append(fqdn)

        fqdn_urls = sorted(url_fqdn_neis.keys())
        url_fqdn_neis = [url_fqdn_neis[url] for url in fqdn_urls]
        url_fqdn_neis = np.array([np.array(neis) for neis in url_fqdn_neis], dtype=object)

        np.save(os.path.join(part_dir, "nei_f.npy"), url_fqdn_neis)

        fqnd_url_len = [len(i) for i in url_fqdn_neis]
        print(max(fqnd_url_len), min(fqnd_url_len), np.mean(fqnd_url_len))

        # ------------ fqdn_registered_domain_neis ------------
        fqdn_registered_domain_relations = np.genfromtxt(os.path.join(part_dir, "fr.txt"))

        fqdn_registered_domain_neis = {}
        for fqdn_registered_domain_relation in fqdn_registered_domain_relations:
            fqdn = fqdn_registered_domain_relation[0]
            registered_domain = fqdn_registered_domain_relation[1]
            fqdn_registered_domain_neis.setdefault(fqdn, []).append(registered_domain)

        fqdn_registered_domains = sorted(fqdn_registered_domain_neis.keys())
        fqdn_registered_domain_neis = [
            fqdn_registered_domain_neis[fqdn] for fqdn in fqdn_registered_domains
        ]
        fqdn_registered_domain_neis = np.array(
            [np.array(neis) for neis in fqdn_registered_domain_neis]
        )

        np.save(os.path.join(part_dir, "nei_r.npy"), fqdn_registered_domain_neis)

        fqdn_registered_domain_len = [len(i) for i in fqdn_registered_domain_neis]
        print(
            max(fqdn_registered_domain_len),
            min(fqdn_registered_domain_len),
            np.mean(fqdn_registered_domain_len),
        )

        # ------------ url_word_neis ------------
        word_url_relations = np.genfromtxt(os.path.join(part_dir, "wu.txt"))

        url_word_neis = {}
        for word_url_relation in word_url_relations:
            word = word_url_relation[0]
            url = word_url_relation[1]
            url_word_neis.setdefault(url, []).append(word)

        word_urls = sorted(url_word_neis.keys())
        url_word_neis = [url_word_neis[url] for url in word_urls]
        url_word_neis = np.array([np.array(neis) for neis in url_word_neis], dtype=object)

        np.save(os.path.join(part_dir, "nei_w.npy"), url_word_neis)

        url_word_len = [len(i) for i in url_word_neis]
        print(max(url_word_len), min(url_word_len), np.mean(url_word_len))
