import os
import numpy as np
import scipy.sparse as sp
import pandas as pd

####################################################
# Generate meta-path adjacency matrices for each split part.
####################################################

DATASET_ROOT = "dataset"
CURRENT_DIR = os.path.dirname(__file__)
NUM_PARTS = 5

if __name__ == "__main__":

    for part_num in range(NUM_PARTS):
        dataset_dir = os.path.join(
            CURRENT_DIR,
            f"../data/{DATASET_ROOT}/phishscope/phishscope_part{part_num}",
        )
        export_dir = os.path.join(
            CURRENT_DIR,
            f"../data/{DATASET_ROOT}/graphish/graphish_part{part_num}",
        )
        if not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        url_nodes = pd.read_csv(f"{dataset_dir}/url_nodes.csv")
        fqdn_nodes = pd.read_csv(f"{dataset_dir}/fqdn_nodes.csv")
        registered_domain_nodes = pd.read_csv(
            f"{dataset_dir}/registered_domain_nodes.csv"
        )
        word_nodes = pd.read_csv(f"{dataset_dir}/word_nodes.csv")

        url_num = url_nodes.iloc[-1]["url_id"] + 1
        fqdn_num = fqdn_nodes.iloc[-1]["fqdn_id"] + 1
        registered_domain_num = registered_domain_nodes.iloc[-1]["registered_domain_id"] + 1
        word_num = word_nodes.iloc[-1]["word_id"] + 1
        print(part_num, url_num, fqdn_num, registered_domain_num, word_num)

        fqdn_url_relations = np.genfromtxt(f"{export_dir}/fu.txt")
        fqdn_registered_domain_relations = np.genfromtxt(f"{export_dir}/fr.txt")
        word_url_relations = np.genfromtxt(f"{export_dir}/wu.txt")

        fqdn_url_matrix = sp.coo_matrix(
            (
                np.ones(fqdn_url_relations.shape[0]),
                (fqdn_url_relations[:, 0], fqdn_url_relations[:, 1]),
            ),
            shape=(fqdn_num, url_num),
        )
        fqdn_registered_domain_matrix = sp.coo_matrix(
            (
                np.ones(fqdn_registered_domain_relations.shape[0]),
                (
                    fqdn_registered_domain_relations[:, 0],
                    fqdn_registered_domain_relations[:, 1],
                ),
            ),
            shape=(fqdn_num, registered_domain_num),
        )
        word_url_matrix = sp.coo_matrix(
            (
                np.ones(word_url_relations.shape[0]),
                (word_url_relations[:, 0], word_url_relations[:, 1]),
            ),
            shape=(word_num, url_num),
        )

        url_fqdn_url = fqdn_url_matrix.T @ fqdn_url_matrix
        url_fqdn_url = url_fqdn_url > 0
        sp.save_npz(f"{export_dir}/ufu.npz", url_fqdn_url)

        url_fqdn_registered = fqdn_url_matrix.T @ fqdn_registered_domain_matrix
        url_fqdn_registered_fqdn_url = url_fqdn_registered @ url_fqdn_registered.T
        url_fqdn_registered_fqdn_url = url_fqdn_registered_fqdn_url > 0
        sp.save_npz(f"{export_dir}/ufrfu.npz", url_fqdn_registered_fqdn_url)

        url_word_url = word_url_matrix.T @ word_url_matrix
        url_word_url = url_word_url > 0
        sp.save_npz(f"{export_dir}/uwu.npz", url_word_url)
