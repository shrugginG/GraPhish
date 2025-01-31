import os
import pandas as pd
import numpy as np
import random

GRAPHISH_NAME = "graphish_v2"
CURRENT_DIR = os.path.dirname(__file__)


def trans_labels(data_dir, export_dir):

    with open(f"{data_dir}/url_nodes.csv") as f:
        url_nodes = pd.read_csv(f)

    node_labels = []

    for row in url_nodes.iterrows():
        if row[1]["category"] == "benign":
            node_labels.append(0)
        elif row[1]["category"] == "phishy":
            node_labels.append(1)

    label_array = np.array(node_labels, dtype="int32")

    label_npy_file = f"{export_dir}/labels.npy"

    np.save(label_npy_file, label_array)


# NOTE - Deprecated, please use data/neibor_phishing.py instead
def generate_nei_f(data_dir, export_dir):
    nei_f = [[] for _ in range(1018)]
    with open(f"{data_dir}/url_fqdn_relation_edges.csv") as f:
        url_fqdn_relation_edges = pd.read_csv(f)

    url_fqdn_relation_edges["url_id"] = url_fqdn_relation_edges["url_id"] - 1
    url_fqdn_relation_edges["fqdn_id"] = url_fqdn_relation_edges["fqdn_id"] - 1

    for row in url_fqdn_relation_edges.iterrows():
        if row[1]["fqdn_id"] not in nei_f[row[1]["url_id"]]:
            nei_f[row[1]["url_id"]].append(row[1]["fqdn_id"])
        # else:
        #     print("false")

    nei_f = [np.array(nei) for nei in nei_f]

    object_array = np.array(nei_f, dtype="object")

    npy_file = f"{export_dir}/nei_f.npy"

    np.save(npy_file, object_array)


def export_url_fqdn_edge_to_txt(data_dir, export_dir):
    with open(f"{data_dir}/fqdn_url_relations.csv") as f:
        url_fqdn_relation_edges = pd.read_csv(f)

    txt_lines = []
    for row in url_fqdn_relation_edges.iterrows():
        txt_lines.append([row[1]["fqdn_id"], row[1]["url_id"]])

    txt_lines.sort()
    txt_lines = [f"{txt_line[0]}\t{txt_line[1]}\n" for txt_line in txt_lines]
    with open(f"{export_dir}/fu.txt", "w") as f:
        f.writelines(txt_lines)


def export_fqdn_registered_domain_edge_to_txt(data_dir, export_dir):
    # with open(f"{data_dir}/fqdn_registered_domain_relation_edges.csv") as f:
    with open(f"{data_dir}/fqdn_registered_domain_relations.csv") as f:
        fqdn_registered_domain_relation_edges = pd.read_csv(f)

    txt_lines = []
    for row in fqdn_registered_domain_relation_edges.iterrows():
        txt_lines.append([row[1]["fqdn_id"], row[1]["registered_domain_id"]])

    txt_lines.sort()
    txt_lines = [f"{txt_line[0]}\t{txt_line[1]}\n" for txt_line in txt_lines]
    with open(f"{export_dir}/fr.txt", "w") as f:
        f.writelines(txt_lines)


def export_url_word_edge_to_txt(data_dir, export_dir):
    with open(f"{data_dir}/word_url_relations.csv") as f:
        url_word_relation_edges = pd.read_csv(f)

    txt_lines = []
    for row in url_word_relation_edges.iterrows():
        txt_lines.append([row[1]["word_id"], row[1]["url_id"]])

    txt_lines.sort()
    txt_lines = [f"{txt_line[0]}\t{txt_line[1]}\n" for txt_line in txt_lines]
    with open(f"{export_dir}/wu.txt", "w") as f:
        f.writelines(txt_lines)


# Divide the data into training, testing andvalidation sets
def divide_datasets(data_dir, export_dir, ratio=[20, 40, 60]):
    with open(f"{data_dir}/url_nodes.csv") as f:
        url_nodes = pd.read_csv(f)

    benign_start_index = 0
    for node in url_nodes.iterrows():
        if node[1]["category"] == "phishy":
            benign_end_index = node[1]["url_id"] - 1
            break
    phishy_start_index = benign_end_index + 1
    phishy_end_index = url_nodes.iloc[-1]["url_id"]

    for num in ratio:
        all_nodes = list(range(phishy_end_index + 1))

        # Get training nodes
        benign_train_index = random.sample(
            range(benign_start_index, benign_end_index + 1), num
        )
        phishy_train_index = random.sample(
            range(phishy_start_index, phishy_end_index + 1), num
        )
        train_index = sorted(benign_train_index + phishy_train_index)

        # Get remaining nodes after removing training nodes
        remain_nodes_benign = [
            n
            for n in range(benign_start_index, benign_end_index + 1)
            if n not in benign_train_index
        ]
        remain_nodes_phishy = [
            n
            for n in range(phishy_start_index, phishy_end_index + 1)
            if n not in phishy_train_index
        ]

        # Get test nodes - 500 from each category
        test_benign = random.sample(remain_nodes_benign, 500)
        test_phishy = random.sample(remain_nodes_phishy, 500)
        test_index = sorted(test_benign + test_phishy)

        # Get remaining nodes after removing test nodes
        remain_nodes_benign = [n for n in remain_nodes_benign if n not in test_benign]
        remain_nodes_phishy = [n for n in remain_nodes_phishy if n not in test_phishy]

        # Get validation nodes - 500 from each category
        val_benign = random.sample(remain_nodes_benign, 500)
        val_phishy = random.sample(remain_nodes_phishy, 500)
        val_index = sorted(val_benign + val_phishy)

        # Save indices
        train_index = np.array(train_index, dtype="int64")
        np.save(f"{export_dir}/train_{num}.npy", train_index)
        test_index = np.array(test_index, dtype="int64")
        np.save(f"{export_dir}/test_{num}.npy", test_index)
        val_index = np.array(val_index, dtype="int64")
        np.save(f"{export_dir}/val_{num}.npy", val_index)


def divide_datasets_1(data_dir, export_dir, ratio=[20, 40, 60]):
    with open(f"{data_dir}/url_nodes.csv") as f:
        url_nodes = pd.read_csv(f)

    grouped = url_nodes.groupby(["source", "registered_domain", "category"])
    # grouped = url_nodes.groupby(['registered_domain', 'category'])

    for num in ratio:
        benign_train_index = []
        phishy_train_index = []

        for name, group in grouped:
            category = name[2]
            group_size = len(group)
            sample_size = min(group_size, num // 10)

            sampled_indices = random.sample(list(group["url_id"]), sample_size)

            if category == "benign":
                benign_train_index.extend(sampled_indices)
            elif category == "phishy":
                phishy_train_index.extend(sampled_indices)

        train_index = sorted(benign_train_index + phishy_train_index)

        # Get labels for training set
        train_labels = []
        for idx in train_index:
            label = (
                1
                if url_nodes.loc[url_nodes["url_id"] == idx, "category"].iloc[0]
                == "phishy"
                else 0
            )
            train_labels.append(label)

        train_labels = np.array(train_labels, dtype="int64")

        # 获取剩余节点
        all_benign = url_nodes[url_nodes["category"] == "benign"]["url_id"].tolist()
        all_phishy = url_nodes[url_nodes["category"] == "phishy"]["url_id"].tolist()

        remain_nodes_benign = [n for n in all_benign if n not in benign_train_index]
        remain_nodes_phishy = [n for n in all_phishy if n not in phishy_train_index]

        # 获取测试节点 - 每个类别500个
        test_benign = random.sample(
            remain_nodes_benign, min(500, len(remain_nodes_benign))
        )
        test_phishy = random.sample(
            remain_nodes_phishy, min(500, len(remain_nodes_phishy))
        )
        test_index = sorted(test_benign + test_phishy)

        # 获取剩余节点
        remain_nodes_benign = [n for n in remain_nodes_benign if n not in test_benign]
        remain_nodes_phishy = [n for n in remain_nodes_phishy if n not in test_phishy]

        # 获取验证节点 - 每个类别500个
        val_benign = random.sample(
            remain_nodes_benign, min(500, len(remain_nodes_benign))
        )
        val_phishy = random.sample(
            remain_nodes_phishy, min(500, len(remain_nodes_phishy))
        )
        val_index = sorted(val_benign + val_phishy)

        # 保存索引
        train_index = np.array(train_index, dtype="int64")
        np.save(f"{export_dir}/train_{num}.npy", train_index)
        np.save(f"{export_dir}/train_labels_{num}.npy", train_labels)
        test_index = np.array(test_index, dtype="int64")
        np.save(f"{export_dir}/test_{num}.npy", test_index)
        val_index = np.array(val_index, dtype="int64")
        np.save(f"{export_dir}/val_{num}.npy", val_index)


if __name__ == "__main__":

    ratios = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    for ratio in ratios:

        dataset_types = [
            "anchor_href",
            "urls_in_html",
            "urls_in_har",
            "all_related",
        ]

        for dataset_type in dataset_types:

            dataset_dir = os.path.join(
                CURRENT_DIR,
                f"../data/graphish_v2/{dataset_type}/{dataset_type}_phishscope_{ratio}_part",
            )
            export_dir = os.path.join(
                CURRENT_DIR,
                f"../data/graphish_v2/{dataset_type}/{dataset_type}_graphish_{ratio}_part",
            )
            if not os.path.exists(export_dir):
                os.makedirs(export_dir, exist_ok=True)

            trans_labels(dataset_dir, export_dir)

            export_url_fqdn_edge_to_txt(dataset_dir, export_dir)

            export_fqdn_registered_domain_edge_to_txt(dataset_dir, export_dir)

            export_url_word_edge_to_txt(dataset_dir, export_dir)

            # export_fqdn_ip_edge_to_txt(data_dir, export_dir)

            divide_datasets_1(dataset_dir, export_dir)
