import csv
import os
import numpy as np
import scipy.sparse as sp
import pandas as pd

####################################################
# This tool is to generate meta-path based adjacency
# matrix given original links.
####################################################

DATASET_ROOT = "dataset"
CURRENT_DIR = os.path.dirname(__file__)
NUM_PARTS = 5

def mask_graphish_rdomain_edge(
    fqdn_url_matrix, fqdn_registered_domain_matrix, ufrfu_adjacency_matrix
):
    ufrfu_edge_cnt = ufrfu_adjacency_matrix.nnz

    url_registered_matrix = fqdn_url_matrix.T @ fqdn_registered_domain_matrix
    # url_registered_url = url_registered @ url_registered.T
    # url_registered_url = url_registered_url > 0

    registered_domain_tranco_path = os.path.join(
        CURRENT_DIR, "../data/tranco/YX64G.csv"
    )
    with open(registered_domain_tranco_path, "r") as file:
        csv_reader = csv.reader(file)
        tranco_rank = {row[1]: int(row[0]) for row in csv_reader}

    max_rank = max(tranco_rank.values()) + 1
    # Get ranks for each registered domain, using max_rank + 1 for domains not in tranco list
    registered_domain_ranks = [
        tranco_rank.get(domain, max_rank)
        for domain in registered_domain_nodes["registered_domain"]
    ]

    # Sort registered domain indices by rank (ascending)
    sorted_registered_domain_indices = np.argsort(registered_domain_ranks)

    # Initialize variables
    target_edge_cnt = ufrfu_edge_cnt // 2
    current_rdomain_idx = 0
    valid_rdomain_indices = np.arange(len(registered_domain_ranks))

    while True:
        # Remove the registered domain node with highest rank (lowest rank number)
        valid_rdomain_indices = np.delete(
            valid_rdomain_indices,
            np.where(
                valid_rdomain_indices
                == sorted_registered_domain_indices[current_rdomain_idx]
            ),
        )
        print(f"Removing registered domain {registered_domain_nodes['registered_domain'][sorted_registered_domain_indices[current_rdomain_idx]]}")

        filtered_url_registered_matrix = url_registered_matrix.tocsc()[
            :, valid_rdomain_indices
        ]
        filtered_url_registered_matrix = filtered_url_registered_matrix.tocoo()

        # Calculate new ufrfu matrix
        masked_ufrfu = filtered_url_registered_matrix @ filtered_url_registered_matrix.T
        masked_ufrfu = masked_ufrfu > 0

        print(f"Current edge ratio: {masked_ufrfu.nnz/ufrfu_edge_cnt}")
        # Check if edge count is below target
        if masked_ufrfu.nnz <= target_edge_cnt:
            break

        current_rdomain_idx += 1

    final_masked_ufrfu = ufrfu_adjacency_matrix - masked_ufrfu
    # print(f"Original ufrfu edge count: {ufrfu_edge_cnt}")
    # print(f"Masked ufrfu edge count: {masked_ufrfu.nnz}")
    # print(f"Final masked ufrfu edge count: {final_masked_ufrfu.nnz}")
    # print(f"Final masked ufrfu edge ratio: {final_masked_ufrfu.nnz/ufrfu_edge_cnt}")

    return final_masked_ufrfu


def get_uwu_adjacency_matrix(word_url_matrix):
    url_word_url = word_url_matrix.T @ word_url_matrix
    url_word_url = url_word_url > 0
    uwu_edge_cnt = url_word_url.nnz
    print(f"Loaded url_word_url matrix statistics:")
    print(f"Shape: {url_word_url.shape}")
    print(f"Number of edges: {uwu_edge_cnt}")
    print(
        f"Density: {uwu_edge_cnt / (url_word_url.shape[0] * url_word_url.shape[1]):.4f}"
    )
    return url_word_url, uwu_edge_cnt


def get_ufrfu_adjacency_matrix(fqdn_url_matrix, fqdn_registered_domain_matrix):
    url_rdomain = fqdn_url_matrix.T @ fqdn_registered_domain_matrix
    url_rdomain_url = url_rdomain @ url_rdomain.T
    url_rdomain_url = url_rdomain_url > 0

    ufrfu_edge_cnt = url_rdomain_url.nnz
    print(f"Loaded url_rdomain_url matrix statistics:")
    print(f"Shape: {url_rdomain_url.shape}")
    print(f"Number of edges: {ufrfu_edge_cnt}")
    print(
        f"Density: {ufrfu_edge_cnt / (url_rdomain_url.shape[0] * url_rdomain_url.shape[1]):.4f}"
    )
    return url_rdomain_url, ufrfu_edge_cnt


def mask_graphish_word_edge(word_url_matrix, uwu_edge_cnt):
    # Calculate degree of each word node
    word_degrees = np.array(word_url_matrix.sum(axis=1)).flatten()

    # Sort word nodes by degree in descending order
    sorted_word_indices = np.argsort(-word_degrees)

    # Initialize variables
    target_edge_cnt = uwu_edge_cnt // 2
    current_word_idx = 0
    valid_word_indices = np.arange(word_degrees.shape[0])

    while True:
        # Get current word degree being removed
        current_word_degree = word_degrees[sorted_word_indices[current_word_idx]]
        print(f"Removing word with degree {current_word_degree}")

        # Remove the word node with highest degree
        valid_word_indices = np.delete(
            valid_word_indices,
            np.where(valid_word_indices == sorted_word_indices[current_word_idx]),
        )

        # Filter word_url_matrix to only keep rows for valid word nodes
        filtered_word_url_matrix = word_url_matrix.tocsr()[valid_word_indices]
        filtered_word_url_matrix = filtered_word_url_matrix.tocoo()

        # Calculate new uwu matrix
        masked_url_word_url = filtered_word_url_matrix.T @ filtered_word_url_matrix
        masked_url_word_url = masked_url_word_url > 0

        print(f"Current edge ratio: {masked_url_word_url.nnz/uwu_edge_cnt}")
        # Check if edge count is below target
        if masked_url_word_url.nnz <= target_edge_cnt:
            break

        current_word_idx += 1

    return masked_url_word_url


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

        # Read CSV files for node info
        url_nodes = pd.read_csv(f"{dataset_dir}/url_nodes.csv")
        fqdn_nodes = pd.read_csv(f"{dataset_dir}/fqdn_nodes.csv")
        registered_domain_nodes = pd.read_csv(
            f"{dataset_dir}/registered_domain_nodes.csv"
        )
        word_nodes = pd.read_csv(f"{dataset_dir}/word_nodes.csv")

        # Extract number of nodes in each category
        url_num = url_nodes.iloc[-1]["url_id"] + 1
        fqdn_num = fqdn_nodes.iloc[-1]["fqdn_id"] + 1
        registered_domain_num = (
            registered_domain_nodes.iloc[-1]["registered_domain_id"] + 1
        )
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

        # # Get ufrfu adjacency matrix
        ufrfu_adjacency_matrix, ufrfu_edge_cnt = get_ufrfu_adjacency_matrix(
            fqdn_url_matrix, fqdn_registered_domain_matrix
        )

        # # Get masked ufrfu adjacency matrix
        masked_ufrfu_adjacency_matrix = mask_graphish_rdomain_edge(
            fqdn_url_matrix, fqdn_registered_domain_matrix, ufrfu_adjacency_matrix
        )
        print(masked_ufrfu_adjacency_matrix.shape)
        sp.save_npz(f"{export_dir}/masked_ufrfu_1.npz", masked_ufrfu_adjacency_matrix)

        # Get uwu adjacency matrix
        uwu_adjacency_matrix, uwu_edge_cnt = get_uwu_adjacency_matrix(
            word_url_matrix
        )

        # Get masked uwu adjacency matrix
        masked_uwu_adjacency_matrix = mask_graphish_word_edge(
            word_url_matrix, uwu_edge_cnt
        )

        # Save uwu adjacency matrix
        # sp.save_npz(f"{export_dir}/uwu.npz", uwu_adjacency_matrix)
        sp.save_npz(f"{export_dir}/masked_uwu_1.npz", masked_uwu_adjacency_matrix)

        # url_fqdn_url = fqdn_url_matrix.T @ fqdn_url_matrix
        # url_fqdn_url = url_fqdn_url == 1
        # url_fqdn_url = url_fqdn_url > 1

        # print(url_fqdn_url)
        # sp.save_npz(f"{export_dir}/masked_ufu_1.npz", url_fqdn_url)
        # print(f"Loaded url_fqdn_url matrix statistics:")
        # print(f"Shape: {url_fqdn_url.shape}")
        # print(f"Number of edges: {url_fqdn_url.nnz}")
        # print(
        #     f"Density: {url_fqdn_url.nnz / (url_fqdn_url.shape[0] * url_fqdn_url.shape[1]):.4f}"
        # )
        # # Load and print statistics about the saved matrix
        # loaded_ufu = sp.load_npz(f"{export_dir}/masked_ufu_1.npz")
        # print(loaded_ufu.shape)


        # url_fqdn_registered = fqdn_url_matrix.T @ fqdn_registered_domain_matrix
        # url_fqdn_registered_fqdn_url = url_fqdn_registered @ url_fqdn_registered.T
        # url_fqdn_registered_fqdn_url = url_fqdn_registered_fqdn_url > 0
        # print(f"Loaded url_fqdn_registered_fqdn_url matrix statistics:")
        # print(f"Shape: {url_fqdn_registered_fqdn_url.shape}")
        # print(f"Number of edges: {url_fqdn_registered_fqdn_url.nnz}")
        # print(
        #     f"Density: {url_fqdn_registered_fqdn_url.nnz / (url_fqdn_registered_fqdn_url.shape[0] * url_fqdn_registered_fqdn_url.shape[1]):.4f}"
        # )

        # ------------------- Mask fqdn nodes -------------------
        # Calculate degree of each fqdn node
        # fqdn_degrees = np.array(fqdn_url_matrix.sum(axis=1)).flatten()
        # # Get indices of fqdn nodes with degree < 3
        # valid_fqdn_indices = np.where(fqdn_degrees < 600)[0]
        # # Filter fqdn_url_matrix to only keep rows for valid fqdn nodes
        # # Convert to CSR format for efficient row indexing
        # fqdn_url_matrix_csr = fqdn_url_matrix.tocsr()
        # # Now we can safely use row indexing
        # filtered_fqdn_url_matrix = fqdn_url_matrix_csr[valid_fqdn_indices]
        # # Convert back to COO format to match original
        # filtered_fqdn_url_matrix = filtered_fqdn_url_matrix.tocoo()
        # # Recalculate url_fqdn_url with filtered matrix
        # masked_url_fqdn_url = filtered_fqdn_url_matrix.T @ filtered_fqdn_url_matrix
        # # Convert to binary adjacency matrix
        # masked_url_fqdn_url = masked_url_fqdn_url > 0

        # print(f"Loaded masked_url_fqdn_url matrix statistics:")
        # print(f"Shape: {masked_url_fqdn_url.shape}")
        # print(f"Number of edges: {masked_url_fqdn_url.nnz}")
        # print(
        #     f"Density: {masked_url_fqdn_url.nnz / (masked_url_fqdn_url.shape[0] * masked_url_fqdn_url.shape[1]):.4f}"
        # )
        # Save filtered matrix
        # sp.save_npz(f"{export_dir}/masked_ufu.npz", masked_url_fqdn_url)
        # print(f"Saved masked_url_fqdn_url matrix to {export_dir}/masked_ufu.npz")

        # # ------------------- Mask registered domain nodes -------------------
        # registered_domain_degrees = np.array(
        #     fqdn_registered_domain_matrix.sum(axis=1)
        # ).flatten()
        # valid_registered_domain_indices = np.where(registered_domain_degrees < 2)[
        #     0
        # ]
        # filtered_fqdn_registered_domain_matrix = (
        #     fqdn_registered_domain_matrix.tocsr()[valid_registered_domain_indices]
        # )
        # filtered_fqdn_registered_domain_matrix = (
        #     filtered_fqdn_registered_domain_matrix.tocoo()
        # )
        # masked_url_fqdn_registered = (
        #     fqdn_url_matrix.T @ filtered_fqdn_registered_domain_matrix
        # )
        # masked_url_fqdn_registered_fqdn_url = (
        #     masked_url_fqdn_registered @ masked_url_fqdn_registered.T
        # )
        # masked_url_fqdn_registered_fqdn_url = masked_url_fqdn_registered_fqdn_url > 0

        # print(f"Loaded masked_url_fqdn_registered_fqdn_url matrix statistics:")
        # print(f"Shape: {masked_url_fqdn_registered_fqdn_url.shape}")
        # print(f"Number of edges: {masked_url_fqdn_registered_fqdn_url.nnz}")
        # print(
        #     f"Density: {masked_url_fqdn_registered_fqdn_url.nnz / (masked_url_fqdn_registered_fqdn_url.shape[0] * masked_url_fqdn_registered_fqdn_url.shape[1]):.4f}"
        # )

        # ------------------- Mask word nodes -------------------
        # word_degrees = np.array(word_url_matrix.sum(axis=1)).flatten()
        # valid_word_indices = np.where(word_degrees < 100)[0]
        # filtered_word_url_matrix = word_url_matrix.tocsr()[valid_word_indices]
        # filtered_word_url_matrix = filtered_word_url_matrix.tocoo()
        # masked_url_word_url = filtered_word_url_matrix.T @ filtered_word_url_matrix
        # masked_url_word_url = masked_url_word_url > 0

        # print(f"Loaded masked_url_word_url matrix statistics:")
        # print(f"Shape: {masked_url_word_url.shape}")
        # print(f"Number of edges: {masked_url_word_url.nnz}")
        # print(
        #     f"Density: {masked_url_word_url.nnz / (masked_url_word_url.shape[0] * masked_url_word_url.shape[1]):.4f}"
        # )
        # # Save filtered matrix
        # sp.save_npz(f"{export_dir}/masked_uwu.npz", masked_url_word_url)
        # print(f"Saved masked_url_word_url matrix to {export_dir}/masked_uwu.npz")

        # # Load and print statistics about the saved matrix
        # # loaded_ufu = sp.load_npz(f"{export_di}/ufu.npz")

        # break
