import argparse

DEFAULT_SPLITS = ["phishscope10", "phishscope20", "phishscope30"]

DATASET_ARGS = {
    "graphish_part0": {
        "type_num": [7039, 31164, 16597, 13157],
        "nei_num": 2,
        "n_labels": 2,
    },
    "graphish_part1": {
        "type_num": [7033, 31317, 16679, 13280],
        "nei_num": 2,
        "n_labels": 2,
    },
    "graphish_part2": {
        "type_num": [7024, 31569, 17502, 13427],
        "nei_num": 2,
        "n_labels": 2,
    },
    "graphish_part3": {
        "type_num": [7015, 37781, 19791, 13067],
        "nei_num": 2,
        "n_labels": 2,
    },
    "graphish_part4": {
        "type_num": [7006, 32589, 16706, 13246],
        "nei_num": 2,
        "n_labels": 2,
    },
}


def build_args():
    parser = argparse.ArgumentParser(description="GraPhish classification pipeline")
    parser.add_argument("--dataset", type=str, default="graphish_part0")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Split names that map to train_<split>.npy, val_<split>.npy and test_<split>.npy.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    # Encoder / decoder
    parser.add_argument("--num_heads", type=int, default=8, help="hidden attention heads")
    parser.add_argument(
        "--num_out_heads", type=int, default=1, help="output attention heads"
    )
    parser.add_argument("--num_layers", type=int, default=2, help="hidden layers")
    parser.add_argument(
        "--residual", action="store_true", default=True, help="residual connection flag"
    )
    parser.add_argument("--feat_drop", type=float, default=0.4, help="feature dropout")
    parser.add_argument("--attn_drop", type=float, default=0, help="attention dropout")
    parser.add_argument("--norm", type=str, default="batchnorm")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--negative_slope",
        type=float,
        default=0.2,
        help="negative slope of leaky relu for GAT",
    )
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument(
        "--feat_mask_rate",
        type=str,
        default="0.2,0.005,0.5",
        help="Feature mask rate schedule.",
    )
    parser.add_argument(
        "--replace_rate",
        type=float,
        default=0.3,
        help="Ratio of nodes replaced by random nodes during masking.",
    )
    parser.add_argument(
        "--leave_unchanged",
        type=float,
        default=0.1,
        help="Ratio of nodes left unchanged but still reconstructed.",
    )

    parser.add_argument("--encoder", type=str, default="han")
    parser.add_argument("--decoder", type=str, default="han")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=3, help="pow index for sce loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", action="store_true", default=True)
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.99,
        help="Decay factor for the ExponentialLR scheduler.",
    )

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--mae_epochs", type=int, default=10000)

    # Evaluation head
    parser.add_argument("--eva_lr", type=float, default=0.01)
    parser.add_argument("--eva_wd", type=float, default=0, help="weight decay")

    # Training process
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--l2_coef", type=float, default=0)

    # Metapath2Vec
    parser.add_argument(
        "--use_mp2vec_feat_pred",
        type=bool,
        default=True,
        help="Enable mp2vec feature regularization.",
    )
    parser.add_argument("--mps_lr", type=float, default=0.005, help="mp2vec learning rate")
    parser.add_argument("--mps_embedding_dim", type=int, default=64)
    parser.add_argument("--mps_walk_length", type=int, default=10)
    parser.add_argument("--mps_context_size", type=int, default=5)
    parser.add_argument("--mps_walks_per_node", type=int, default=3)
    parser.add_argument("--mps_num_negative_samples", type=int, default=3)
    parser.add_argument("--mps_batch_size", type=int, default=256)
    parser.add_argument("--mps_epoch", type=int, default=20)
    parser.add_argument("--mp2vec_feat_pred_loss_weight", type=float, default=0.5)
    parser.add_argument(
        "--mp2vec_feat_alpha_l",
        type=float,
        default=2,
        help="Pow index for SCE loss in mp2vec feature prediction.",
    )
    parser.add_argument(
        "--mp2vec_feat_drop", type=float, default=0.2, help="mp2vec feature dropout"
    )

    parser.add_argument(
        "--use_cfg", action="store_true", help="Read hyperparameters from configs.yml"
    )

    # Meta-path edge reconstruction
    parser.add_argument(
        "--use_mp_edge_recon",
        type=bool,
        default=True,
        help="Enable meta-path edge reconstruction.",
    )
    parser.add_argument("--mp_edge_recon_loss_weight", type=float, default=0.5)
    parser.add_argument(
        "--mp_edge_mask_rate",
        type=str,
        default="0.5",
        help="Mask rate schedule for meta-path edges.",
    )
    parser.add_argument(
        "--mp_edge_alpha_l",
        type=float,
        default=3,
        help="Pow index for SCE loss in edge reconstruction.",
    )

    # Label information
    parser.add_argument(
        "--n_labels",
        type=int,
        default=2,
        help="Number of labels in the target classification task.",
    )

    args, _ = parser.parse_known_args()
    if args.dataset not in DATASET_ARGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    for key, value in DATASET_ARGS[args.dataset].items():
        setattr(args, key, value)
    return args
