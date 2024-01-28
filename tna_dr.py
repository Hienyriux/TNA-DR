# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import zipfile
import argparse

import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

def set_all_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def mean_and_std(stats):
    mean = sum(stats) / len(stats)
    
    std = 0
    for stat in stats:
        std += (stat - mean) * (stat - mean)
    std /= len(stats)
    std = std ** 0.5
    
    return mean * 100, std * 100

def calc_metrics(y_pred, y_true):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    
    auroc = metrics.roc_auc_score(y_true, y_pred)
    aupr = metrics.average_precision_score(y_true, y_pred)
    
    return auroc, aupr

def get_num_drugs(dataset_name):
    num_drugs_dict = {
        "Fdataset" : 593,
        "Cdataset" : 663,
        "lrssl" : 763,
        "repodb" : 1572
    }
    
    return num_drugs_dict[dataset_name]

def get_num_diseases(dataset_name):
    num_diseases_dict = {
        "Fdataset" : 313,
        "Cdataset" : 409,
        "lrssl" : 681,
        "repodb" : 2074
    }
    
    return num_diseases_dict[dataset_name]

class LinkPredictor(nn.Module):
    def __init__(
        self,
        num_drugs,
        num_diseases,
        num_fp_feats=512,
        hidden_dim=256,
        pre_mp_dropout=0.8,
        post_mp_dropout=0.5,
        pred_dropout=0.8):
        
        super().__init__()
        
        self.drug_fc = nn.Linear(num_fp_feats, hidden_dim)
        self.adj_fc = nn.Linear(num_diseases, hidden_dim)
        
        self.disease_pre_mp = nn.Sequential(
            nn.Linear(num_diseases, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(pre_mp_dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.disease_post_mp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(post_mp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.pred_mlp = nn.Sequential(
            nn.Dropout(pred_dropout),
            nn.Linear(hidden_dim, num_diseases)
        )
    
    def do_mp(self, x, adj):
        x_mp = self.disease_pre_mp(x)
        adj_mp = F.normalize(adj, p=1, dim=-1)
        x_mp = adj_mp @ x_mp        
        x_mp = self.disease_post_mp(x_mp)
        return x_mp
    
    def forward(self, drug_sim, fp, adj, disease_sim):
        x_drug = self.drug_fc(fp)
        
        x_adj = self.adj_fc(adj)
        x_drug = x_drug + x_adj
        
        x_mp = self.do_mp(disease_sim, adj)
        x_drug = x_drug + x_mp
        
        out = self.pred_mlp(x_drug)
        
        return out, x_drug

def get_tail_nodes(args, adj, aug_type, tail_threshold, tag):    
    if aug_type.startswith("drug"):
        num_nodes = args.num_drugs
        degs = adj.sum(dim=1)
    else:
        num_nodes = args.num_diseases
        degs = adj.sum(dim=0)
    
    if tail_threshold > 0:
        tail_num = int(tail_threshold * num_nodes)
        _, tail_nodes = degs.topk(tail_num, largest=False)
    else:
        tail_nodes = (degs == 0).nonzero().squeeze(-1)
    
    tail_nodes = tail_nodes.tolist()
    
    print(
        f"{tag} Head Num: {num_nodes - len(tail_nodes)}, "
        f"Tail Num: {len(tail_nodes)}"
    )
    
    return tail_nodes

def get_knn_loss(
    drug_sim, disease_sim, adj, adj_out,
    tail_nodes, aug_type, num_neighbors):
    
    if aug_type == "drug":
        tgt_sim = drug_sim[tail_nodes]
        num_neighbors = min(num_neighbors, args.num_drugs - 1)
    else:
        tgt_sim = disease_sim[tail_nodes]
        num_neighbors = min(num_neighbors, args.num_diseases - 1)
    
    # (num_tail_nodes, num_nodes)
    
    diag_idx = torch.arange(tgt_sim.size(0))
    tgt_sim[diag_idx, diag_idx] = 0
    
    y_src = adj_out
    y_tgt = adj
    
    if aug_type == "disease":
        y_src = y_src.t()
        y_tgt = y_tgt.t()
    
    # (num_nodes, num_labels)
    
    y_src = y_src[tail_nodes]
    # (num_tail_nodes, num_labels)
    
    y_src = y_src.flatten()
    # (num_tail_nodes * num_labels, )
    
    num_tail_nodes = len(tail_nodes)
    
    if num_neighbors > 0:
        topk_ind = tgt_sim.topk(k=num_neighbors, dim=-1)[1]
        topk_ind = topk_ind.flatten().cpu()
        # (num_tail_nodes * k, )
        
        src_ind = torch.arange(num_tail_nodes).unsqueeze(-1)
        src_ind = src_ind.tile(1, num_neighbors).flatten()
        # (num_tail_nodes * k, )
        
        y_tgt = y_tgt[topk_ind]
        # (num_tail_nodes * k, num_labels)
        
        y_tgt = y_tgt.reshape(num_tail_nodes, num_neighbors, -1)
        # (num_tail_nodes, k, num_labels)
        
        tgt_sim = tgt_sim[src_ind, topk_ind].reshape(num_tail_nodes, -1)
        # (num_tail_nodes, k)
    
    else:
        # 将所有节点都作为邻居
        y_tgt = y_tgt.unsqueeze(0).tile(num_tail_nodes, 1, 1)
        # (num_tail_nodes, num_nodes, num_labels)

    # 无需调整
    tau = 1.0
    
    tgt_sim = (tgt_sim / tau).softmax(dim=-1)
    # (num_tail_nodes, k)
    tgt_sim = tgt_sim.unsqueeze(-1)
    # (num_tail_nodes, k, 1)
    
    y_aggr = (tgt_sim * y_tgt).sum(dim=1)
    # (num_tail_nodes, num_labels)
    y_aggr = y_aggr.clamp(0, 1).flatten()
    # (num_tail_nodes * num_labels, )
    
    knn_loss = F.binary_cross_entropy_with_logits(y_src, y_aggr)
    
    return knn_loss

def get_contra_loss(drug_sim, disease_sim, src, tail_nodes, aug_type):
    if aug_type.startswith("drug"):
        tgt_sim = drug_sim[tail_nodes]
    else:
        tgt_sim = disease_sim[tail_nodes]
    
    x = F.normalize(src, dim=-1)
    src_sim = x @ x.t()
    src_sim = src_sim[tail_nodes]
    
    contra_loss = F.mse_loss(src_sim, tgt_sim)
    
    return contra_loss

def get_data(args):
    dataset_name = args.dataset_name
    mode = args.mode
    device = args.device
    
    dir_path = f"dataset/{dataset_name}"
    
    load_path = f"{dir_path}/split_{mode}"

    if args.num_folds == 10:
        load_path = f"{load_path}_10_10.zip"
    else:
        load_path = f"{load_path}_1_5.zip"
    
    with zipfile.ZipFile(load_path, "r") as zipf:
        filename = load_path.split("/")[-1].replace(".zip", ".json")
        with zipf.open(filename, "r") as f:
            split_data_all = json.load(f)
    
    load_path = f"{dir_path}/drug_sim.pt"
    drug_sim = torch.load(load_path).to(device)
    
    load_path = f"{dir_path}/disease_sim.pt"
    disease_sim = torch.load(load_path).to(device)

    load_path = f"{dir_path}/fp.pt"
    fp = torch.load(load_path).to(device)
    
    return split_data_all, drug_sim, disease_sim, fp

def get_model(args):
    model = LinkPredictor(
        args.num_drugs,
        args.num_diseases,
        512,
        args.hidden_dim,
        args.pre_mp_dropout,
        args.post_mp_dropout,
        args.pred_dropout,
    ).to(args.device)

    return model

def get_coords(args, split_data, split_name, device):
    if split_name in split_data:
        return torch.LongTensor(split_data[split_name]).to(device).t()
    
    adj_tmp = torch.ones(args.num_drugs, args.num_diseases)

    for cur_split in split_data.values():
        drugs, diseases = map(list, zip(*cur_split))
        adj_tmp[drugs, diseases] = 0

    return adj_tmp.nonzero().to(device).t()

def train_valid(args, split_data_all, drug_sim, disease_sim, fp):
    criterion = nn.BCEWithLogitsLoss()
    
    cv_ind = [int(cur.strip()) for cur in args.cv_ind.split(",")]
    
    for cv_idx in cv_ind:
        for fold in range(args.num_folds):
            set_all_seeds(args.seed)
            
            print(f"Cross Validation ID: {cv_idx}, Fold ID: {fold}")
            
            split_data = split_data_all[cv_idx][fold]
            
            train_pos = get_coords(args, split_data, "train_pos", args.device)
            train_neg = get_coords(args, split_data, "train_neg", args.device)
            valid_pos = get_coords(args, split_data, "valid_pos", args.device)
            valid_neg = get_coords(args, split_data, "valid_neg", args.device)
            
            # 除了train_pos, 其他位置全部置零
            adj = torch.zeros(args.num_drugs, args.num_diseases).to(args.device)
            adj[train_pos[0], train_pos[1]] = 1
            
            train_ind = torch.cat([train_pos, train_neg], dim=-1)
            valid_ind = torch.cat([valid_pos, valid_neg], dim=-1)
            
            y_true_train = torch.Tensor(
                [1] * train_pos.size(1) + [0] * train_neg.size(1)
            ).to(args.device)
            
            y_true_valid = torch.Tensor(
                [1] * valid_pos.size(1) + [0] * valid_neg.size(1)
            ).to(args.device)
            
            tail_nodes_knn = get_tail_nodes(
                args, adj, args.aug_type_knn,
                args.tail_threshold_knn, "kNN"
            )

            tail_nodes_contra = get_tail_nodes(
                args, adj, args.aug_type_contra,
                args.tail_threshold_contra, "Contra"
            )
            
            model = get_model(args)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            
            for epoch in range(args.num_epochs):
                model.train()

                adj_out, x_drug = model(drug_sim, fp, adj, disease_sim)
                
                y_pred_train = adj_out[train_ind[0], train_ind[1]]

                cls_loss = criterion(y_pred_train, y_true_train)

                loss = cls_loss
                
                if args.lamb_knn > 0:
                    knn_loss = get_knn_loss(
                        drug_sim, disease_sim, adj, adj_out,
                        tail_nodes_knn, args.aug_type_knn, args.num_neighbors
                    )
                    
                    loss = loss + args.lamb_knn * knn_loss
                
                if args.lamb_contra > 0:
                    if args.aug_type_contra == "drug_rep":
                        src = x_drug
                    elif args.aug_type_contra == "drug_pred":
                        src = adj_out
                    else:
                        src = adj_out.t()
                    
                    contra_loss = get_contra_loss(
                        drug_sim, disease_sim, src,
                        tail_nodes_contra, args.aug_type_contra
                    )

                    loss = loss + args.lamb_contra * contra_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                model.eval()
                
                with torch.no_grad():
                    adj_out, _ = model(drug_sim, fp, adj, disease_sim)
                
                y_pred_train = adj_out[train_ind[0], train_ind[1]]
                y_pred_valid = adj_out[valid_ind[0], valid_ind[1]]
                
                y_pred_train = y_pred_train.sigmoid().detach().cpu()
                y_pred_valid = y_pred_valid.sigmoid().detach().cpu()
                
                train_auroc, train_aupr = calc_metrics(
                    y_pred_train, y_true_train.cpu()
                )
                valid_auroc, valid_aupr = calc_metrics(
                    y_pred_valid, y_true_valid.cpu()
                )

                out_text = (
                    f"\rEpoch: {epoch}, Loss: {loss.item():.4f}, "
                    f"CLS Loss: {cls_loss.item():.4f}, "
                )
                
                if args.lamb_knn:
                    out_text += f"kNN Loss: {knn_loss.item():.4f}, "
                if args.lamb_contra:
                    out_text += f"Contra Loss: {contra_loss.item():.4f}, "
                
                out_text += (
                    f"Train AUROC: {train_auroc:.4f}, "
                    f"Train AUPR: {train_aupr:.4f}, "
                    f"Valid AUROC: {valid_auroc:.4f}, "
                    f"Valid AUPR: {valid_aupr:.4f}"
                )
                
                sys.stdout.write(out_text)
                sys.stdout.flush()
            
            print("\n")

def train(args, split_data_all, drug_sim, disease_sim, fp):
    os.makedirs("model", exist_ok=True)
    
    criterion = nn.BCEWithLogitsLoss()
    
    cv_ind = [int(cur.strip()) for cur in args.cv_ind.split(",")]
    
    for cv_idx in cv_ind:
        for fold in range(args.num_folds):
            set_all_seeds(args.seed)
            
            print(f"Cross Validation ID: {cv_idx}, Fold ID: {fold}")
            
            split_data = split_data_all[cv_idx][fold]
            
            train_pos = get_coords(args, split_data, "train_pos", args.device)
            train_neg = get_coords(args, split_data, "train_neg", args.device)
            
            # 除了train_pos, 其他位置全部置零
            adj = torch.zeros(args.num_drugs, args.num_diseases).to(args.device)
            adj[train_pos[0], train_pos[1]] = 1
            
            train_ind = torch.cat([train_pos, train_neg], dim=-1)
            
            y_true_train = torch.Tensor(
                [1] * train_pos.size(1) + [0] * train_neg.size(1)
            ).to(args.device)
            
            tail_nodes_knn = get_tail_nodes(
                args, adj, args.aug_type_knn,
                args.tail_threshold_knn, "kNN"
            )

            tail_nodes_contra = get_tail_nodes(
                args, adj, args.aug_type_contra,
                args.tail_threshold_contra, "Contra"
            )
            
            model = get_model(args)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            
            for epoch in range(args.num_epochs):
                model.train()

                adj_out, x_drug = model(drug_sim, fp, adj, disease_sim)
                
                y_pred_train = adj_out[train_ind[0], train_ind[1]]
                
                cls_loss = criterion(y_pred_train, y_true_train)

                loss = cls_loss
                
                if args.lamb_knn > 0:
                    knn_loss = get_knn_loss(
                        drug_sim, disease_sim, adj, adj_out,
                        tail_nodes_knn, args.aug_type_knn, args.num_neighbors
                    )
                    
                    loss = loss + args.lamb_knn * knn_loss
                
                if args.lamb_contra > 0:
                    if args.aug_type_contra == "drug_rep":
                        src = x_drug
                    elif args.aug_type_contra == "drug_pred":
                        src = adj_out
                    else:
                        src = adj_out.t()
                    
                    contra_loss = get_contra_loss(
                        drug_sim, disease_sim, src,
                        tail_nodes_contra, args.aug_type_contra
                    )

                    loss = loss + args.lamb_contra * contra_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                model.eval()
                
                with torch.no_grad():
                    adj_out, _ = model(drug_sim, fp, adj, disease_sim)
                
                y_pred_train = adj_out[train_ind[0], train_ind[1]]                
                y_pred_train = y_pred_train.sigmoid().detach().cpu()
                
                train_auroc, train_aupr = calc_metrics(
                    y_pred_train, y_true_train.cpu()
                )
                
                out_text = (
                    f"\rEpoch: {epoch}, Loss: {loss.item():.4f}, "
                    f"CLS Loss: {cls_loss.item():.4f}, "
                )
                
                if args.lamb_knn:
                    out_text += f"kNN Loss: {knn_loss.item():.4f}, "
                if args.lamb_contra:
                    out_text += f"Contra Loss: {contra_loss.item():.4f}, "

                out_text += (
                    f"Train AUROC: {train_auroc:.4f}, "
                    f"Train AUPR: {train_aupr:.4f}"
                )
                
                sys.stdout.write(out_text)
                sys.stdout.flush()
            
            save_path = f"model/{args.dataset_name}_{cv_idx}_{fold}.pt"
            torch.save(model.state_dict(), save_path)

            print("\n")

def test(args, split_data_all, drug_sim, disease_sim, fp):
    criterion = nn.BCEWithLogitsLoss()
    
    cv_ind = [int(cur.strip()) for cur in args.cv_ind.split(",")]
    
    for cv_idx in cv_ind:
        stats_auroc = []
        stats_aupr = []
        
        for fold in range(args.num_folds):
            set_all_seeds(args.seed)
            
            print(f"Cross Validation ID: {cv_idx}, Fold ID: {fold}")
            
            split_data = split_data_all[cv_idx][fold]
            
            train_pos = get_coords(args, split_data, "train_pos", args.device)
            test_pos = get_coords(args, split_data, "test_pos", args.device)
            test_neg = get_coords(args, split_data, "test_neg", args.device)
            
            # 除了train_pos, 其他位置全部置零
            adj = torch.zeros(args.num_drugs, args.num_diseases).to(args.device)
            adj[train_pos[0], train_pos[1]] = 1
            
            test_ind = torch.cat([test_pos, test_neg], dim=-1)
            
            y_true_test = torch.Tensor(
                [1] * test_pos.size(1) + [0] * test_neg.size(1)
            ).to(args.device)
            
            model = get_model(args)
            
            load_path = f"model/{args.dataset_name}_{cv_idx}_{fold}.pt"
            
            model.load_state_dict(torch.load(
                load_path, map_location=args.device
            ))
            
            model.eval()
            
            with torch.no_grad():
                adj_out, _ = model(drug_sim, fp, adj, disease_sim)
            
            y_pred_test = adj_out[test_ind[0], test_ind[1]]
            y_pred_test = y_pred_test.sigmoid().detach().cpu()
            y_true_test = y_true_test.cpu()
            
            test_auroc, test_aupr = calc_metrics(y_pred_test, y_true_test)
            
            stats_auroc.append(test_auroc)
            stats_aupr.append(test_aupr)
            
            print(f"Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}")
            print()

        auroc_mean, auroc_std = mean_and_std(stats_auroc)
        aupr_mean, aupr_std = mean_and_std(stats_aupr)

        print(
            f"Test AUROC Mean±Std: {auroc_mean:.2f}±{auroc_std:.2f}, "
            f"Test AUPR Mean±Std: {aupr_mean:.2f}±{aupr_std:.2f}"
        )

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name", type=str,
        choices=["Fdataset", "Cdataset", "lrssl", "repodb"],
        default="Fdataset"
    )

    parser.add_argument(
        "--mode", type=str,
        choices=["train_valid", "train", "test"],
        default="train"
    )
    
    parser.add_argument(
        "--cv_ind", type=str,
        default="0,1,2,3,4,5,6,7,8,9"
    )

    parser.add_argument(
        "--num_folds", type=int,
        choices=[5, 10],
        default=10
    )

    parser.add_argument("--num_epochs", type=int, default=1600)
    parser.add_argument("--lr", type=float, default=0.001)
    
    parser.add_argument("--hidden_dim", type=int, default=256)
    
    parser.add_argument("--pre_mp_dropout", type=float, default=0.8)
    parser.add_argument("--post_mp_dropout", type=float, default=0.5)
    parser.add_argument("--pred_dropout", type=float, default=0.8)

    parser.add_argument(
        "--aug_type_knn", type=str,
        choices=["drug", "disease"],
        default="disease"
    )
    
    parser.add_argument(
        "--aug_type_contra", type=str,
        choices=["drug_rep", "drug_pred", "disease"],
        default="drug_rep"
    )

    parser.add_argument("--tail_threshold_knn", type=float, default=0.0)
    parser.add_argument("--tail_threshold_contra", type=float, default=0.0)

    parser.add_argument("--num_neighbors", type=int, default=5)
    
    parser.add_argument("--lamb_knn", type=float, default=0.2)
    parser.add_argument("--lamb_contra", type=float, default=0.2)
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()

    num_drugs = get_num_drugs(args.dataset_name)
    num_diseases = get_num_diseases(args.dataset_name)

    args.num_drugs = num_drugs
    args.num_diseases = num_diseases

    return args

if __name__ == "__main__":
    args = get_args()
    
    split_data_all, drug_sim, disease_sim, fp = get_data(args)
    
    if args.mode == "train_valid":
        train_valid(args, split_data_all, drug_sim, disease_sim, fp)
    elif args.mode == "train":
        train(args, split_data_all, drug_sim, disease_sim, fp)
    else:
        test(args, split_data_all, drug_sim, disease_sim, fp)
