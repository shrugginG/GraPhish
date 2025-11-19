import datetime
import warnings
from pathlib import Path

import torch
from torch_geometric.nn.models import MetaPath2Vec

from phishgmae.models.edcoder import PhisHMAE_Model
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
            url_feat,  # Target node feats
            metapath_adjacency_matrices,  # List of metapath adjacency matrices
            node_labels,  # target node labels
        classifier_train_index,  # index of training nodes, [[20 * nb_classes], [40 * nb_classes], [80 * nb_classes]]
        classifier_val_index,  # index of val nodes
        classifier_test_index,  # index of test nodes
        masked_metapath_adjacency_matrices,
    ),
    phish_graph,
    mp2vec_metapaths,
    ) = load_data(args.dataset, args.splits, args.type_num)
    nb_classes = node_labels.shape[-1]
    url_feat_dim = url_feat.shape[1]

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
            mp2vec_url_feat = metapath_model("url").detach()

            # free up memory
            del metapath_model
            if args.device.type == "cuda":
                mp2vec_url_feat = mp2vec_url_feat.cpu()
                mp2vec_url_feats.append(mp2vec_url_feat)
                torch.cuda.empty_cache()
        mp2vec_feat = torch.cat(
            [torch.FloatTensor(preprocess_features(feat)) for feat in mp2vec_url_feats],
            dim=1,
        )
        url_feat = torch.hstack([url_feat, mp2vec_feat])

    # Use the two meta-path adjacency matrices that align with the GraPhish setting.
    metapath_adjacency_matrices = [
        metapath_adjacency_matrices[1],
        metapath_adjacency_matrices[2],
    ]
    metapath_num = len(metapath_adjacency_matrices)

    # model
    graphish_model = PhisHMAE_Model(
        args, metapath_num, url_feat_dim, args.mps_embedding_dim * len(mp2vec_metapaths)
    )

    optimizer = torch.optim.Adam(
        graphish_model.parameters(), lr=args.lr, weight_decay=args.l2_coef
    )
    # scheduler
    if args.scheduler:
        print("--- Use schedular ---")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.scheduler_gamma,  # default: 0.99
        )
    # if args.scheduler:
    #     print("--- 使用余弦退火热重启调度器 ---")
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #         optimizer, T_0=10, T_mult=2, eta_min=1e-6
    #     )
    else:
        scheduler = None

    graphish_model.to(args.device)
    url_feat = url_feat.to(args.device)
    metapath_adjacency_matrices = [
        metapath_adjacency_matrice.to(args.device)
        for metapath_adjacency_matrice in metapath_adjacency_matrices
    ]
    masked_metapath_adjacency_matrices = [
        metapath_adjacency_matrice.to(args.device)
        for metapath_adjacency_matrice in masked_metapath_adjacency_matrices
    ]
    node_labels = node_labels.to(args.device)
    classifier_train_index = [i.to(args.device) for i in classifier_train_index]
    classifier_val_index = [i.to(args.device) for i in classifier_val_index]
    classifier_test_index = [i.to(args.device) for i in classifier_test_index]

    cnt_wait = 0
    best_loss = 1e9
    best_epoch = 0

    starttime = datetime.datetime.now()
    best_model_state_dict = None
    for epoch in range(args.mae_epochs):  # default: 10000
        graphish_model.train()
        optimizer.zero_grad()
        loss, loss_item = graphish_model(
            url_feat,
            metapath_adjacency_matrices,
            masked_metapath_adjacency_matrices,
            epoch=epoch,
        )
        print(
            f"Epoch: {epoch}, loss: {loss_item}, lr: {optimizer.param_groups[0]['lr']:.6f}"
        )
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            cnt_wait = 0
            best_model_state_dict = graphish_model.state_dict()
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print("Early stopping!")
            break
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    print("The best epoch is: ", best_epoch)
    graphish_model.load_state_dict(best_model_state_dict)
    graphish_model.eval()
    embeds = graphish_model.get_embeds(url_feat, metapath_adjacency_matrices)
    
    embedding_dir = Path(__file__).resolve().parent / "data" / "urlnode_embedding"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeds.cpu(), embedding_dir / f"{args.dataset}.pt")

    # Compute wall-clock time for producing the embeddings
    embedding_endtime = datetime.datetime.now()
    embedding_time = (embedding_endtime - starttime).seconds
    print("Time to obtain embeddings: ", embedding_time, "s")
    
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
        args = load_best_configs(args, "configs.yml")

    print(args)
    main(args)

