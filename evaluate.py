import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import softmax

NUM_PARTS = 5
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(CURRENT_DIR, "data/urlnode_embedding/")
DATASET_PATH = os.path.join(CURRENT_DIR, "data/dataset/graphish/")
PHISHSCOPE_SPLITS = ["phishscope10", "phishscope20", "phishscope30"]
RESULT_PATH = os.path.join(CURRENT_DIR, "data/results/")
LOG_PATH = os.path.join(CURRENT_DIR, "data/logs/")
LOG_FILE = os.path.join(LOG_PATH, "evaluate.log")

os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running evaluation on device: {device}")


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.2, activation: str = "relu"
    ):
        super().__init__()

        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.1)
        elif activation == "elu":
            act_fn = nn.ELU()
        else:
            act_fn = nn.ReLU()

        layers: List[nn.Module] = []

        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(act_fn)
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(dropout))

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(act_fn)
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def encode_onehot(labels: np.ndarray) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def evaluate(
    embeds: torch.Tensor,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
    label: torch.Tensor,
    nb_classes: int,
    device: torch.device,
    model_type: str = "mlp",
    lr: float = 0.01,
    wd: float = 0.0001,
    hidden_dims: List[int] = [256, 128],
    dropout: float = 0.3,
    patience: int = 20,
    max_iters: int = 200,
    split_name: str | None = None,
) -> Tuple[float, float, float, float, float]:
    """Evaluate embeddings with a shallow classifier."""
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)

    accs: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    auc_score_list: List[float] = []

    best_run_predictions = None
    best_run_probabilities = None
    best_run_accuracy = -1

    for i in range(50):
        if model_type == "mlp":
            if i == 0:
                print(f"Run {i+1}/50: using MLP classifier")
            model = MLP(hid_units, nb_classes, hidden_dims=hidden_dims, dropout=dropout)
        else:
            if i == 0:
                print(f"Run {i+1}/50: using LogReg classifier")
            from phishgmae.utils.logreg import LogReg

            model = LogReg(hid_units, nb_classes)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        model.to(device)

        val_accs: List[float] = []
        test_accs: List[float] = []
        test_precisions: List[float] = []
        test_recalls: List[float] = []
        test_f1s: List[float] = []
        logits_list: List[torch.Tensor] = []

        no_improve = 0
        best_val_acc = 0

        for iter_ in range(max_iters):
            model.train()
            opt.zero_grad()

            logits = model(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                logits = model(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]

            val_accs.append(val_acc.item())

            with torch.no_grad():
                logits = model(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average="macro")

            test_accs.append(test_acc.item())
            test_f1s.append(test_f1_macro)
            test_precisions.append(
                precision_score(test_lbls.cpu(), preds.cpu(), average="macro", zero_division=0)
            )
            test_recalls.append(
                recall_score(test_lbls.cpu(), preds.cpu(), average="macro", zero_division=0)
            )

            logits_list.append(logits)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if i == 0:
                    print(f"Early stopping at iteration {iter_}")
                break

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        precisions.append(test_precisions[max_iter])
        recalls.append(test_recalls[max_iter])
        f1s.append(test_f1s[max_iter])

        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)

        y_true = test_lbls.detach().cpu().numpy()

        if nb_classes <= 2:
            y_score = best_proba[:, 1].detach().cpu().numpy()
            auc_score = roc_auc_score(y_true=y_true, y_score=y_score)
        else:
            auc_score = roc_auc_score(
                y_true=test_lbls.detach().cpu().numpy(),
                y_score=best_proba.detach().cpu().numpy(),
                multi_class="ovr",
            )

        auc_score_list.append(auc_score)

        if test_accs[max_iter] > best_run_accuracy:
            best_run_accuracy = test_accs[max_iter]
            best_run_predictions = torch.argmax(best_logits, dim=1).detach().cpu().numpy()
            best_run_probabilities = best_proba.detach().cpu().numpy()

    if best_run_predictions is not None and split_name:
        true_labels = test_lbls.detach().cpu().numpy()

        filename = f"prediction_results_{split_name}_{model_type}.txt"
        filepath = os.path.join(RESULT_PATH, filename)

        print(f"\nSaving prediction details to: {filepath}")

        with open(filepath, "w") as f:
            f.write("# True label, Predicted label, Predicted Probability\n")

            if nb_classes <= 2:
                for idx in range(len(true_labels)):
                    pred_prob = best_run_probabilities[idx, 1]
                    f.write(
                        f"{true_labels[idx]:.6f}\t{best_run_predictions[idx]:.6f}\t{pred_prob:.6f}\n"
                    )
            else:
                for idx in range(len(true_labels)):
                    pred_class = best_run_predictions[idx]
                    pred_prob = best_run_probabilities[idx, pred_class]
                    f.write(
                        f"{true_labels[idx]:.6f}\t{pred_class:.6f}\t{pred_prob:.6f}\n"
                    )

        print(f"Saved {len(true_labels)} URL predictions")

    print(
        "\t[Classification] Accuracy: [{:.4f}, {:.4f}]  Precision: [{:.4f}, {:.4f}]  "
        "Recall: [{:.4f}, {:.4f}]  F1-score: [{:.4f}, {:.4f}]  AUC: [{:.4f}, {:.4f}]".format(
            np.mean(accs),
            np.std(accs),
            np.mean(precisions),
            np.std(precisions),
            np.mean(recalls),
            np.std(recalls),
            np.mean(f1s),
            np.std(f1s),
            np.mean(auc_score_list),
            np.std(auc_score_list),
        )
    )
    return (
        np.mean(accs),
        np.mean(precisions),
        np.mean(recalls),
        np.mean(f1s),
        np.mean(auc_score_list),
    )


def load_partition_data(split_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load embeddings, labels, and indices for a specific split across all parts."""
    url_embedding_list: List[torch.Tensor] = []
    classifier_train_indices: List[torch.Tensor] = []
    classifier_test_indices: List[torch.Tensor] = []
    classifier_val_indices: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []
    node_counts: List[int] = []

    for i in range(NUM_PARTS):
        print(f"Loading part {i+1}/{NUM_PARTS} ...")

        embedding_path = os.path.join(EMBEDDINGS_PATH, f"graphish_part{i}.pt")
        label_path = os.path.join(DATASET_PATH, f"graphish_part{i}/labels.npy")
        train_indices_path = os.path.join(
            DATASET_PATH, f"graphish_part{i}/train_{split_name}.npy"
        )
        test_indices_path = os.path.join(
            DATASET_PATH, f"graphish_part{i}/test_{split_name}.npy"
        )
        val_indices_path = os.path.join(
            DATASET_PATH, f"graphish_part{i}/val_{split_name}.npy"
        )

        for path in [embedding_path, label_path, train_indices_path, test_indices_path, val_indices_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required file for part {i}: {path}")

        embedding = torch.load(embedding_path)
        url_embedding_list.append(embedding)
        node_counts.append(embedding.shape[0])

        label = np.load(label_path).astype("int32")
        label = torch.FloatTensor(encode_onehot(label))
        label_list.append(label)

        classifier_train_indices.append(torch.LongTensor(np.load(train_indices_path)))
        classifier_test_indices.append(torch.LongTensor(np.load(test_indices_path)))
        classifier_val_indices.append(torch.LongTensor(np.load(val_indices_path)))

    adjusted_train_indices = []
    adjusted_test_indices = []
    adjusted_val_indices = []
    offset = 0

    for i in range(NUM_PARTS):
        adjusted_train_indices.append(classifier_train_indices[i] + offset)
        adjusted_test_indices.append(classifier_test_indices[i] + offset)
        adjusted_val_indices.append(classifier_val_indices[i] + offset)
        offset += node_counts[i]

    combined_embeddings = torch.cat(url_embedding_list, dim=0)
    combined_labels = torch.cat(label_list, dim=0)
    combined_train_indices = torch.cat(adjusted_train_indices, dim=0)
    combined_test_indices = torch.cat(adjusted_test_indices, dim=0)
    combined_val_indices = torch.cat(adjusted_val_indices, dim=0)

    print(f"Combined embeddings shape: {combined_embeddings.shape}")
    print(f"Combined labels shape: {combined_labels.shape}")
    print(f"Training indices: {len(combined_train_indices)}")
    print(f"Test indices: {len(combined_test_indices)}")
    print(f"Validation indices: {len(combined_val_indices)}")

    combined_embeddings = combined_embeddings.to(device)
    combined_labels = combined_labels.to(device)
    combined_train_indices = combined_train_indices.to(device)
    combined_test_indices = combined_test_indices.to(device)
    combined_val_indices = combined_val_indices.to(device)

    return (
        combined_embeddings,
        combined_train_indices,
        combined_val_indices,
        combined_test_indices,
        combined_labels,
    )


def process_split(split_name: str, model_type: str = "mlp") -> Dict[str, float]:
    """Evaluate a specific split across all parts."""
    print(f"\n{'='*80}")
    print(f"Processing split {split_name} (model: {model_type})")
    print(f"{'='*80}")

    (
        combined_embeddings,
        combined_train_indices,
        combined_val_indices,
        combined_test_indices,
        combined_labels,
    ) = load_partition_data(split_name)

    nb_classes = combined_labels.shape[1]
    print(f"Number of classes: {nb_classes}")

    accuracy, precision, recall, f1, auc = evaluate(
        combined_embeddings,
        combined_train_indices,
        combined_val_indices,
        combined_test_indices,
        combined_labels,
        nb_classes,
        device,
        model_type=model_type,
        lr=0.005 if model_type == "mlp" else 0.01,
        wd=0.0001 if model_type == "mlp" else 0.0,
        hidden_dims=[256, 128],
        dropout=0.3,
        patience=15,
        split_name=split_name,
    )

    result = {
        "split": split_name,
        "model": model_type,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }

    print(f"\nSplit {split_name} ({model_type}) evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy-AUC Gap: {auc-accuracy:.4f}")

    return result


def main():
    print("Evaluating merged GraPhish partitions on all PhishScope splits...")

    all_results: Dict[str, Dict[str, float]] = {}
    model_types = ["mlp"]

    for model_type in model_types:
        all_results[model_type] = {}
        for split_name in PHISHSCOPE_SPLITS:
            result = process_split(split_name, model_type=model_type)
            all_results[model_type][split_name] = result

    print("\n\nSummary:")
    print(f"{'-'*130}")
    print(
        f"{'Model':<10} {'Split':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'AUC':<10} {'ACC-AUC Gap':<10}"
    )
    print(f"{'-'*130}")

    for model_type in model_types:
        for split_name in PHISHSCOPE_SPLITS:
            result = all_results[model_type][split_name]
            gap = result["auc"] - result["accuracy"]
            print(
                f"{result['model']:<10} {result['split']:<15} {result['accuracy']:<10.4f} "
                f"{result['precision']:<10.4f} {result['recall']:<10.4f} {result['f1']:<10.4f} "
                f"{result['auc']:<10.4f} {gap:<10.4f}"
            )
        print(f"{'-'*130}")

    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write("Evaluation summary:\n")
        for model_type in model_types:
            for split_name in PHISHSCOPE_SPLITS:
                result = all_results[model_type][split_name]
                log_file.write(
                    f"{result['model']} {result['split']} "
                    f"acc={result['accuracy']:.4f} "
                    f"precision={result['precision']:.4f} "
                    f"recall={result['recall']:.4f} "
                    f"f1={result['f1']:.4f} "
                    f"auc={result['auc']:.4f}\n"
                )

    print(f"\nSaved detailed URL classification information to {RESULT_PATH}")


if __name__ == "__main__":
    main()
