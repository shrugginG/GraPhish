import copy
import datetime
import itertools
import sys
import warnings
import numpy as np

import torch
from torch_geometric.nn.models import MetaPath2Vec

from phishgmae.models.edcoder import PreModel
from phishgmae.utils import (
    evaluate,
    load_best_configs,
    load_data,
    metapath2vec_train,
    preprocess_features,
    set_random_seed,
)
from phishgmae.utils.params import build_args

warnings.filterwarnings("ignore")


def main(args):
    set_random_seed(args.seed)

    # Load data
    (
        (
            node_feats,  # Target node feats
            metapath_adjacency_matrices,  # List of metapath adjacency matrices
            node_labels,  # target node labels
            classifier_train_index,  # index of training nodes, [[20 * nb_classes], [40 * nb_classes], [80 * nb_classes]]
            classifier_val_index,  # index of val nodes
            classifier_test_index,  # index of test nodes
        ),
        phish_graph,
        mp2vec_metapaths,
    ) = load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = node_labels.shape[-1]
    feats_dim_list = [node_feat.shape[1] for node_feat in node_feats]

    metapath_num = len(metapath_adjacency_matrices)  # metapath nums
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", metapath_num)

    if args.use_mp2vec_feat_pred:
        mp2vec_url_feats = []
        assert args.mps_embedding_dim > 0
        for mp2vec_metapath in mp2vec_metapaths:
            print(mp2vec_metapath)
            metapath_model = MetaPath2Vec(
                phish_graph.edge_index_dict,
                args.mps_embedding_dim,
                mp2vec_metapath,
                args.mps_walk_length,
                args.mps_context_size,
                args.mps_walks_per_node,
                args.mps_num_negative_samples,
                sparse=True,
            )
            metapath2vec_train(args, metapath_model, args.mps_epoch, args.device)
            # mp2vec_feat = metapath_model("target").detach()
            mp2vec_url_feat = metapath_model("url").detach()

            # free up memory
            del metapath_model
            if args.device.type == "cuda":
                mp2vec_url_feat = mp2vec_url_feat.cpu()
                mp2vec_url_feats.append(mp2vec_url_feat)
                torch.cuda.empty_cache()
        mp2vec_feat = torch.FloatTensor(preprocess_features(mp2vec_url_feats[0]))
        node_feats[0] = torch.hstack([node_feats[0], mp2vec_feat])

    # model
    focused_feature_dim = feats_dim_list[0]
    model = PreModel(args, metapath_num, focused_feature_dim)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2_coef
    )
    # scheduler
    if args.scheduler:
        print("--- Use schedular ---")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.scheduler_gamma
        )
    else:
        scheduler = None

    model.to(args.device)
    node_feats = [feat.to(args.device) for feat in node_feats]
    metapath_adjacency_matrices = [
        mp.to(args.device) for mp in metapath_adjacency_matrices
    ]
    node_labels = node_labels.to(args.device)
    classifier_train_index = [i.to(args.device) for i in classifier_train_index]
    classifier_val_index = [i.to(args.device) for i in classifier_val_index]
    classifier_test_index = [i.to(args.device) for i in classifier_test_index]

    cnt_wait = 0
    best = 1e9
    best_t = 0

    starttime = datetime.datetime.now()
    best_model_state_dict = None
    for epoch in range(args.mae_epochs):
        model.train()
        optimizer.zero_grad()
        # loss, loss_item = model(node_feats, metapath_adjacency_matrices, nei_index=url_neighs, epoch=epoch)
        loss, loss_item = model(node_feats, metapath_adjacency_matrices, epoch=epoch)
        print(
            f"Epoch: {epoch}, loss: {loss_item}, lr: {optimizer.param_groups[0]['lr']:.6f}"
        )
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            best_model_state_dict = model.state_dict()
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print("Early stopping!")
            break
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    print("The best epoch is: ", best_t)
    model.load_state_dict(best_model_state_dict)
    model.eval()
    # embeds = model.get_embeds(node_feats, metapath_adjacency_matrices, url_neighs)
    embeds = model.get_embeds(node_feats, metapath_adjacency_matrices)
    if args.task == "classification":
        macro_score_list, micro_score_list, auc_score_list = [], [], []
        for i in range(len(classifier_train_index)):
            macro_score, micro_score, auc_score = evaluate(
                embeds,
                classifier_train_index[i],
                classifier_val_index[i],
                classifier_test_index[i],
                node_labels,
                nb_classes,
                args.device,
                args.eva_lr,
                args.eva_wd,
                args.dataset,
                train_index=i,
            )
            macro_score_list.append(macro_score)
            micro_score_list.append(micro_score)
            auc_score_list.append(auc_score)
        # Save results to CSV
        results_dict = {
            # Model args
            "dataset": args.dataset,
            "activation": args.activation,
            "alpha_l": args.alpha_l,
            "attn_drop": args.attn_drop,
            "decoder": args.decoder,
            "encoder": args.encoder,
            "eva_lr": args.eva_lr,
            "eva_wd": args.eva_wd,
            "feat_drop": args.feat_drop,
            "feat_mask_rate": args.feat_mask_rate,
            "hidden_dim": args.hidden_dim,
            "l2_coef": args.l2_coef,
            "leave_unchanged": args.leave_unchanged,
            "loss_fn": args.loss_fn,
            "lr": args.lr,
            "mask_rate": args.mask_rate,
            "mp2vec_feat_alpha_l": args.mp2vec_feat_alpha_l,
            "mp2vec_feat_drop": args.mp2vec_feat_drop,
            "mp2vec_feat_pred_loss_weight": args.mp2vec_feat_pred_loss_weight,
            "mp_edge_alpha_l": args.mp_edge_alpha_l,
            "mp_edge_mask_rate": args.mp_edge_mask_rate,
            "mp_edge_recon_loss_weight": args.mp_edge_recon_loss_weight,
            "nei_num": args.nei_num,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "num_out_heads": args.num_out_heads,
            "optimizer": args.optimizer,
            "patience": args.patience,
            "replace_rate": args.replace_rate,
            "residual": args.residual,
            "scheduler": args.scheduler,
            "scheduler_gamma": args.scheduler_gamma,
            # Results
            "macro_f1": macro_score_list,
            "micro_f1": micro_score_list,
            "auc": auc_score_list,
        }

        import pandas as pd

        results_df = pd.DataFrame([results_dict])

        # Check if file exists to determine if we need to write headers
        import os

        file_exists = os.path.isfile(f"results/{args.dataset}_results.csv")

        # Append results to CSV, only write headers if file doesn't exist
        results_df.to_csv(
            f"./result/{args.dataset}_results.csv",
            mode="a",
            header=not file_exists,
            index=False,
        )

        print(f"Results appended to results/{args.dataset}_results.csv")
    else:
        sys.exit("wrong args.task.")

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")


if __name__ == "__main__":
    args = build_args()
    if torch.cuda.is_available():
        args.device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        args.device = torch.device("cpu")

    if args.use_cfg:
        if args.task == "classification":
            config_file_name = "configs.yml"
        elif args.task == "clustering":
            config_file_name = "clustering_configs.yml"
        else:
            sys.exit(f"No available config file for task: {args.task}")
        args = load_best_configs(args, config_file_name)

    print(args)
    main(args)
