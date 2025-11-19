import os
import pandas as pd
import numpy as np
import random
import math

CURRENT_DIR = os.path.dirname(__file__)

SPLIT_NAME_MAP = {"20": "phishscope10", "40": "phishscope20", "60": "phishscope30"}


def split_suffix(value):
    return SPLIT_NAME_MAP.get(str(value), str(value))


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
        remain_nodes_benign = [
            n for n in remain_nodes_benign if n not in test_benign]
        remain_nodes_phishy = [
            n for n in remain_nodes_phishy if n not in test_phishy]

        # Get validation nodes - 500 from each category
        val_benign = random.sample(remain_nodes_benign, 500)
        val_phishy = random.sample(remain_nodes_phishy, 500)
        val_index = sorted(val_benign + val_phishy)

        # Save indices
        suffix = split_suffix(num)
        train_index = np.array(train_index, dtype="int64")
        np.save(f"{export_dir}/train_{suffix}.npy", train_index)
        test_index = np.array(test_index, dtype="int64")
        np.save(f"{export_dir}/test_{suffix}.npy", test_index)
        val_index = np.array(val_index, dtype="int64")
        np.save(f"{export_dir}/val_{suffix}.npy", val_index)


def divide_datasets_1(data_dir, export_dir, ratio=[20, 40, 60]):

    with open(f"{data_dir}/url_nodes.csv") as f:
        url_nodes = pd.read_csv(f)

    # grouped = url_nodes.groupby(["source", "registered_domain", "category"])
    grouped = url_nodes.groupby(['registered_domain', 'category'])

    for num in ratio:
        benign_train_index = []
        phishy_train_index = []

        for name, group in grouped:
            category = name[1]
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
        all_benign = url_nodes[url_nodes["category"]
                               == "benign"]["url_id"].tolist()
        all_phishy = url_nodes[url_nodes["category"]
                               == "phishy"]["url_id"].tolist()

        remain_nodes_benign = [
            n for n in all_benign if n not in benign_train_index]
        remain_nodes_phishy = [
            n for n in all_phishy if n not in phishy_train_index]

        # 获取测试节点 - 每个类别500个
        test_benign = random.sample(
            remain_nodes_benign, min(500, len(remain_nodes_benign))
        )
        test_phishy = random.sample(
            remain_nodes_phishy, min(500, len(remain_nodes_phishy))
        )
        test_index = sorted(test_benign + test_phishy)

        # 获取剩余节点
        remain_nodes_benign = [
            n for n in remain_nodes_benign if n not in test_benign]
        remain_nodes_phishy = [
            n for n in remain_nodes_phishy if n not in test_phishy]

        # 获取验证节点 - 每个类别500个
        val_benign = random.sample(
            remain_nodes_benign, min(500, len(remain_nodes_benign))
        )
        val_phishy = random.sample(
            remain_nodes_phishy, min(500, len(remain_nodes_phishy))
        )
        val_index = sorted(val_benign + val_phishy)

        # 保存索引
        suffix = split_suffix(num)
        train_index = np.array(train_index, dtype="int64")
        np.save(f"{export_dir}/train_{suffix}.npy", train_index)
        np.save(f"{export_dir}/train_labels_{suffix}.npy", train_labels)
        test_index = np.array(test_index, dtype="int64")
        np.save(f"{export_dir}/test_{suffix}.npy", test_index)
        val_index = np.array(val_index, dtype="int64")
        np.save(f"{export_dir}/val_{suffix}.npy", val_index)


def divide_datasets_2(
    data_dir,
    export_dir,
    percents=(0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    seed=None
):
    """
    从 data_dir/url_nodes.csv 读取数据，按 (registered_domain, category) 分组，
    对于每个分组分别抽取 percents 中指定比例的样本作为训练集。
    - 训练集：分组内按比例抽样，ceil，至少 1 个（若分组非空）
    - 测试集/验证集：每个类别各最多 500 个，保持与训练集、测试集互斥
    - 输出：train_*.npy, train_labels_*.npy, test_*.npy, val_*.npy
            其中 * 为 0p5pct, 1pct, 2pct ... 的安全文件名后缀
    """
    if seed is not None:
        random.seed(seed)

    os.makedirs(export_dir, exist_ok=True)

    # 读取与基本检查
    url_nodes = pd.read_csv(f"{data_dir}/url_nodes.csv")
    required_cols = {"url_id", "registered_domain", "category"}
    missing = required_cols - set(url_nodes.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    # 按域名+类别分组（与原逻辑一致）
    grouped = url_nodes.groupby(['registered_domain', 'category'])

    # 预构造映射，加速打标签
    cat_map = dict(zip(url_nodes["url_id"], url_nodes["category"]))

    # 全量的每类列表（用于后续从中剔除训练/测试）
    all_benign = url_nodes[url_nodes["category"]
                           == "benign"]["url_id"].tolist()
    all_phishy = url_nodes[url_nodes["category"]
                           == "phishy"]["url_id"].tolist()

    # 文件名后缀格式化：0.5 -> 0p5pct, 1 -> 1pct
    def fmt_pct(p):
        s = str(p).rstrip('0').rstrip('.') if isinstance(p, float) else str(p)
        return f"{s.replace('.', 'p')}pct"

    for p in percents:
        suffix = fmt_pct(p)

        benign_train_index = []
        phishy_train_index = []

        frac = float(p) / 100.0  # 百分比 -> 小数

        # ——按组抽样，分组内按比例取样（至多组大小，至少 1 个若组非空）——
        for (domain, category), group in grouped:
            gsize = len(group)
            if gsize == 0:
                continue
            k = max(1, int(math.ceil(gsize * frac)))
            k = min(k, gsize)
            sampled_indices = random.sample(list(group["url_id"]), k)
            if category == "benign":
                benign_train_index.extend(sampled_indices)
            elif category == "phishy":
                phishy_train_index.extend(sampled_indices)

        # 训练索引与标签
        train_index = sorted(benign_train_index + phishy_train_index)
        train_labels = np.array(
            [1 if cat_map[i] == "phishy" else 0 for i in train_index],
            dtype="int64"
        )

        # ——测试集合（每类最多 500）——
        remain_benign = [n for n in all_benign if n not in benign_train_index]
        remain_phishy = [n for n in all_phishy if n not in phishy_train_index]

        test_benign = random.sample(
            remain_benign, min(500, len(remain_benign)))
        test_phishy = random.sample(
            remain_phishy, min(500, len(remain_phishy)))
        test_index = sorted(test_benign + test_phishy)

        # ——验证集合（每类最多 500）——
        remain_benign = [n for n in remain_benign if n not in test_benign]
        remain_phishy = [n for n in remain_phishy if n not in test_phishy]

        val_benign = random.sample(remain_benign, min(500, len(remain_benign)))
        val_phishy = random.sample(remain_phishy, min(500, len(remain_phishy)))
        val_index = sorted(val_benign + val_phishy)

        # ——保存——
        np.save(f"{export_dir}/train_{suffix}.npy",
                np.array(train_index, dtype="int64"))
        np.save(f"{export_dir}/train_labels_{suffix}.npy", train_labels)
        np.save(f"{export_dir}/test_{suffix}.npy",
                np.array(test_index, dtype="int64"))
        np.save(f"{export_dir}/val_{suffix}.npy",
                np.array(val_index, dtype="int64"))


def add_symmetric_noise(labels, noise_rate, seed=None):
    """
    Add symmetric noise to labels.

    Args:
        labels: Original labels (0 for benign, 1 for phishy)
        noise_rate: Noise rate (0.0 to 1.0)
        seed: Random seed

    Returns:
        Noisy labels
    """
    if seed is not None:
        np.random.seed(seed)

    labels = np.array(labels, copy=True)
    n_samples = len(labels)
    n_noisy = int(n_samples * noise_rate)

    # Randomly select samples to flip
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    # Flip labels
    labels[noisy_indices] = 1 - labels[noisy_indices]

    return labels


def add_class_dependent_noise(labels, rp2b, rb2p, seed=None):
    """
    Add class-dependent (asymmetric) noise to labels.

    Args:
        labels: Original labels (0 for benign, 1 for phishy)
        rp2b: Rate of flipping phishy (1) to benign (0)
        rb2p: Rate of flipping benign (0) to phishy (1)
        seed: Random seed

    Returns:
        Noisy labels
    """
    if seed is not None:
        np.random.seed(seed)

    labels = np.array(labels, copy=True)

    # Get indices for each class
    phishy_indices = np.where(labels == 1)[0]
    benign_indices = np.where(labels == 0)[0]

    # Flip phishy to benign
    n_p2b = int(len(phishy_indices) * rp2b)
    if n_p2b > 0:
        flip_p2b = np.random.choice(phishy_indices, n_p2b, replace=False)
        labels[flip_p2b] = 0

    # Flip benign to phishy
    n_b2p = int(len(benign_indices) * rb2p)
    if n_b2p > 0:
        flip_b2p = np.random.choice(benign_indices, n_b2p, replace=False)
        labels[flip_b2p] = 1

    return labels


def divide_datasets_3(
    data_dir,
    export_dir,
    percents=(0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    seed=None
):
    """
    从 data_dir/url_nodes.csv 读取数据，按 (registered_domain, category) 分组，
    对于每个分组分别抽取 percents 中指定比例的样本作为训练集。
    同时为每个训练集生成不同类型和比例的噪声标签。

    - 训练集：分组内按比例抽样，ceil，至少 1 个（若分组非空）
    - 测试集/验证集：每个类别各最多 500 个，保持与训练集、测试集互斥
    - 噪声类型：
      1. 对称噪声（Symmetric）：噪声率 5%, 10%, 20%, 30%, 40%
      2. 非对称噪声（Class-Dependent）：rp→b ∈ {10%, 20%, 30%}, rb→p ∈ {2%, 5%, 10%}
    - 输出：train_*.npy, train_labels_*.npy, test_*.npy, val_*.npy
           train_labels_{suffix}_{noise_type}_{noise_ratio}.npy
           其中 * 为 0p5pct, 1pct, 2pct ... 的安全文件名后缀
    """
    if seed is not None:
        random.seed(seed)

    os.makedirs(export_dir, exist_ok=True)

    # 读取与基本检查
    url_nodes = pd.read_csv(f"{data_dir}/url_nodes.csv")
    required_cols = {"url_id", "registered_domain", "category"}
    missing = required_cols - set(url_nodes.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    # 按域名+类别分组（与原逻辑一致）
    grouped = url_nodes.groupby(['registered_domain', 'category'])

    # 预构造映射，加速打标签
    cat_map = dict(zip(url_nodes["url_id"], url_nodes["category"]))

    # 全量的每类列表（用于后续从中剔除训练/测试）
    all_benign = url_nodes[url_nodes["category"]
                           == "benign"]["url_id"].tolist()
    all_phishy = url_nodes[url_nodes["category"]
                           == "phishy"]["url_id"].tolist()

    # 文件名后缀格式化：0.5 -> 0p5pct, 1 -> 1pct
    def fmt_pct(p):
        s = str(p).rstrip('0').rstrip('.') if isinstance(p, float) else str(p)
        return f"{s.replace('.', 'p')}pct"

    for p in percents:
        suffix = fmt_pct(p)

        benign_train_index = []
        phishy_train_index = []

        frac = float(p) / 100.0  # 百分比 -> 小数

        # ——按组抽样，分组内按比例取样（至多组大小，至少 1 个若组非空）——
        for (domain, category), group in grouped:
            gsize = len(group)
            if gsize == 0:
                continue
            k = max(1, int(math.ceil(gsize * frac)))
            k = min(k, gsize)
            sampled_indices = random.sample(list(group["url_id"]), k)
            if category == "benign":
                benign_train_index.extend(sampled_indices)
            elif category == "phishy":
                phishy_train_index.extend(sampled_indices)

        # 训练索引与标签
        train_index = sorted(benign_train_index + phishy_train_index)
        train_labels = np.array(
            [1 if cat_map[i] == "phishy" else 0 for i in train_index],
            dtype="int64"
        )

        # ——测试集合（每类最多 500）——
        remain_benign = [n for n in all_benign if n not in benign_train_index]
        remain_phishy = [n for n in all_phishy if n not in phishy_train_index]

        test_benign = random.sample(
            remain_benign, min(500, len(remain_benign)))
        test_phishy = random.sample(
            remain_phishy, min(500, len(remain_phishy)))
        test_index = sorted(test_benign + test_phishy)

        # ——验证集合（每类最多 500）——
        remain_benign = [n for n in remain_benign if n not in test_benign]
        remain_phishy = [n for n in remain_phishy if n not in test_phishy]

        val_benign = random.sample(remain_benign, min(500, len(remain_benign)))
        val_phishy = random.sample(remain_phishy, min(500, len(remain_phishy)))
        val_index = sorted(val_benign + val_phishy)

        # ——保存原始数据——
        np.save(f"{export_dir}/train_{suffix}.npy",
                np.array(train_index, dtype="int64"))
        np.save(f"{export_dir}/train_labels_{suffix}.npy", train_labels)
        np.save(f"{export_dir}/test_{suffix}.npy",
                np.array(test_index, dtype="int64"))
        np.save(f"{export_dir}/val_{suffix}.npy",
                np.array(val_index, dtype="int64"))

        # ——生成噪声标签——
        # 对称噪声 (Symmetric)
        symmetric_rates = [5, 10, 20, 30, 40]  # 噪声率百分比
        for noise_rate in symmetric_rates:
            noisy_labels = add_symmetric_noise(
                train_labels, noise_rate / 100.0, seed)
            noise_suffix = f"symmetric_{noise_rate}p"
            filename = f"train_labels_{suffix}_{noise_suffix}.npy"
            np.save(f"{export_dir}/{filename}", noisy_labels)

        # 非对称噪声 (Class-Dependent)
        # rp→b: phishy to benign rates, rb→p: benign to phishy rates
        class_dependent_configs = [
            (10, 2),   # 10% p→b, 2% b→p
            (10, 5),   # 10% p→b, 5% b→p
            (10, 10),  # 10% p→b, 10% b→p
            (20, 2),   # 20% p→b, 2% b→p
            (20, 5),   # 20% p→b, 5% b→p
            (20, 10),  # 20% p→b, 10% b→p
            (30, 2),   # 30% p→b, 2% b→p
            (30, 5),   # 30% p→b, 5% b→p
            (30, 10),  # 30% p→b, 10% b→p
        ]

        for rp2b, rb2p in class_dependent_configs:
            noisy_labels = add_class_dependent_noise(
                train_labels, rp2b / 100.0, rb2p / 100.0, seed)
            noise_suffix = f"classdep_{rp2b}p2b_{rb2p}b2p"
            filename = f"train_labels_{suffix}_{noise_suffix}.npy"
            np.save(f"{export_dir}/{filename}", noisy_labels)


if __name__ == "__main__":

    NUM_PARTS = 5
    dataset = "dataset"

    for part_num in range(NUM_PARTS):
        dataset_dir = os.path.join(
            CURRENT_DIR,
            f"../data/{dataset}/phishscope/phishscope_part{part_num}",
        )
        export_dir = os.path.join(
            CURRENT_DIR,
            f"../data/{dataset}/graphish/graphish_part{part_num}",
        )
        if not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        divide_datasets_3(dataset_dir, export_dir)
