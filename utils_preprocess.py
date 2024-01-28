# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import random
import zipfile

import numpy as np

import scipy.io as scio
import scipy.sparse as sparse
from scipy.spatial.distance import cdist

from sklearn.model_selection import KFold

import torch

from rdkit import Chem

def process_dataset(dataset_name):
    dir_path = f"dataset/{dataset_name}"
    os.makedirs(dir_path, exist_ok=True)
    
    load_path = f"dataset/raw_data/{dataset_name}.mat"
    data = scio.loadmat(load_path)
    
    drug_dict = data["Wrname"][ : , 0]
    drug_dict = {cur.item() : i for i, cur in enumerate(drug_dict)}
    
    disease_dict = data["Wdname"][ : , 0]
    disease_dict = {cur.item() : i for i, cur in enumerate(disease_dict)}

    save_path = f"{dir_path}/drug_dict.json"    
    with open(save_path, "w") as f:
        json.dump(drug_dict, f, indent=2)

    save_path = f"{dir_path}/disease_dict.json"
    with open(save_path, "w") as f:
        json.dump(disease_dict, f, indent=2)
    
    drug_sim = torch.from_numpy(data["drug"].astype(np.float32))

    save_path = f"dataset/{dataset_name}/drug_sim.pt"
    torch.save(drug_sim, save_path)
    
    print(f"Drugs: {drug_sim.size(0)}")
    
    disease_sim = torch.from_numpy(data["disease"].astype(np.float32))

    save_path = f"dataset/{dataset_name}/disease_sim.pt"
    torch.save(disease_sim, save_path)
    
    print(f"Diseases: {disease_sim.size(0)}")
    
    adj_full = data["didr"].astype(np.int32).T
    adj_full = sparse.csr_matrix(adj_full)

    save_path = f"dataset/{dataset_name}/adj_full.npz"
    sparse.save_npz(save_path, adj_full)

def get_sim_or_adj(entity_dict, load_path):
    with open(load_path, "r") as f:
        lines = f.readlines()

    header = list(filter(None, lines[0][ : -1].split("\t")))
    
    data = []
    row_name_list = []
    
    for line in lines[1 : ]:
        parts = line[ : -1].split("\t")
        parts = list(filter(None, parts))
        
        data.append(list(map(float, parts[1 : ])))
        row_name_list.append(parts[0])
    
    data = torch.Tensor(data)
    
    return data

def process_lrssl():
    load_path = "dataset/raw_data/lrssl_simmat_dc_chemical.txt"
    with open(load_path, "r") as f:
        lines = f.readlines()
    
    drug_dict = {
        line[ : -1].split("\t")[0] : i for i, line in enumerate(lines[1 : ])
    }

    save_path = "dataset/lrssl/drug_dict.json"
    with open(save_path, "w") as f:
        json.dump(drug_dict, f, indent=2)

    load_path = "dataset/raw_data/lrssl_simmat_dg.txt"
    with open(load_path, "r") as f:
        lines = f.readlines()
    
    disease_dict = {
        line[ : -1].split("\t")[0] : i for i, line in enumerate(lines[1 : ])
    }

    save_path = "dataset/lrssl/disease_dict.json"
    with open(save_path, "w") as f:
        json.dump(disease_dict, f, indent=2)
    
    chemical_sim = get_sim_or_adj(
        drug_dict, "dataset/raw_data/lrssl_simmat_dc_chemical.txt"
    )

    domain_sim = get_sim_or_adj(
        drug_dict, "dataset/raw_data/lrssl_simmat_dc_domain.txt"
    )

    go_sim = get_sim_or_adj(
        drug_dict, "dataset/raw_data/lrssl_simmat_dc_go.txt"
    )
    
    drug_sim = (chemical_sim + domain_sim + go_sim) / 3
    print(f"Drugs: {drug_sim.size(0)}")
    
    save_path = "dataset/lrssl/drug_sim.pt"
    torch.save(drug_sim, save_path)
    
    disease_sim = get_sim_or_adj(
        disease_dict, "dataset/raw_data/lrssl_simmat_dg.txt"
    )
    print(f"Diseases: {disease_sim.size(0)}")

    save_path = "dataset/lrssl/disease_sim.pt"
    torch.save(disease_sim, save_path)
    
    adj_full = get_sim_or_adj(
        disease_dict, "dataset/raw_data/lrssl_admat_dgc.txt"
    )

    adj_full = adj_full.numpy().astype(np.int32)
    adj_full = sparse.csr_matrix(adj_full)
    
    save_path = "dataset/lrssl/adj_full.npz"
    sparse.save_npz(save_path, adj_full)

def process_repodb():
    load_path = "dataset/raw_data/repodb_full.csv"
    
    with open(load_path, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)[1 : ]

    drug_dict = set()
    disease_dict = set()

    drug_id_to_name = {}
    disease_id_to_name = {}

    status_dict = set()
    
    for parts in lines:
        drug_name, drug_id, disease_name, disease_id = parts[ : 4]
        status = parts[5]
        
        drug_dict.add(drug_id)
        drug_id_to_name[drug_id] = drug_name

        disease_dict.add(disease_id)
        disease_id_to_name[disease_id] = disease_name

        status_dict.add(status)

    drug_dict = sorted(list(drug_dict))
    disease_dict = sorted(list(disease_dict))
    status_dict = sorted(list(status_dict))
    
    drug_dict = {cur : i for i, cur in enumerate(drug_dict)}
    disease_dict = {cur : i for i, cur in enumerate(disease_dict)}
    status_dict = {cur : i for i, cur in enumerate(status_dict)}
    
    drug_id_to_name = dict(sorted(list(drug_id_to_name.items())))
    disease_id_to_name = dict(sorted(list(disease_id_to_name.items())))
    
    num_drugs = len(drug_dict)
    num_diseases = len(disease_dict)
    
    print(f"Drugs: {num_drugs}")
    print(f"Diseases: {num_diseases}")
    print(f"Status: {len(status_dict)}")
        
    save_path = "dataset/repodb/drug_dict.json"
    with open(save_path, "w") as f:
        json.dump(drug_dict, f, indent=2)
    
    save_path = "dataset/repodb/disease_dict.json"
    with open(save_path, "w") as f:
        json.dump(disease_dict, f, indent=2)

    save_path = "dataset/repodb/status_dict.json"
    with open(save_path, "w") as f:
        json.dump(status_dict, f, indent=2)

    save_path = "dataset/repodb/drug_id_to_name.json"
    with open(save_path, "w") as f:
        json.dump(drug_id_to_name, f, indent=2)
    
    save_path = "dataset/repodb/disease_id_to_name.json"
    with open(save_path, "w") as f:
        json.dump(disease_id_to_name, f, indent=2)
    
    pairs = []
    
    for parts in lines:
        drug_id = parts[1]
        disease_id = parts[3]
        status = parts[5]
        
        pairs.append((
            drug_dict[drug_id],
            disease_dict[disease_id],
            status_dict[status]
        ))

    print(f"Pairs: {len(pairs)}")
    
    save_path = "dataset/repodb/pairs.json"
    with open(save_path, "w") as f:
        json.dump(pairs, f)

def get_rdkfp(dataset_name, minPath=1, maxPath=6, fpSize=512, nBitsPerHash=2):
    load_path = f"dataset/{dataset_name}/smiles.json"
    with open(load_path, "r") as f:
        smiles_list = json.load(f)
    
    rdkfp = []

    is_incomplete = False
    
    for i, smiles in enumerate(smiles_list):
        sys.stdout.write(f"\r{i} / {len(smiles_list)}")
        sys.stdout.flush()

        if len(smiles) == 0:
            rdkfp.append(torch.full((fpSize, ), float("nan")))
            is_incomplete = True
            continue
        
        mol = Chem.MolFromSmiles(smiles)

        obj = Chem.RDKFingerprint(
            mol,
            minPath=minPath,
            maxPath=maxPath,
            fpSize=fpSize,
            nBitsPerHash=nBitsPerHash
        )

        fp_str = obj.ToBitString()
        fp_vec = list(map(int, fp_str))
        
        rdkfp.append(torch.Tensor(fp_vec))

    print()
    
    rdkfp = torch.stack(rdkfp)

    if not is_incomplete:
        save_path = f"dataset/{dataset_name}/fp.pt"
    else:
        save_path = f"dataset/{dataset_name}/fp_incomplete.pt"
    
    torch.save(rdkfp, save_path)

def impute_fp(dataset_name):
    torch.manual_seed(0)

    load_path = f"dataset/{dataset_name}/fp_incomplete.pt"
    fp_raw = torch.load(load_path)
    
    fp_pure = fp_raw[~fp_raw[ : , 0].isnan()]
    num_feats = fp_pure.size(-1)
    
    feat_probs = fp_pure.sum(dim=0)
    mean_num_ones = int(fp_pure.sum(dim=-1).mean().item())
    print(f"Mean Number of \"1\"s in FP: {mean_num_ones}")
    
    fp_full = []
    
    for fp in fp_raw:
        if not fp.isnan().any():
            fp_full.append(fp)
            continue
        
        sampled_feats = torch.multinomial(
            feat_probs, mean_num_ones, replacement=False
        )
        
        sampled_fp = torch.zeros(num_feats)
        sampled_fp[sampled_feats] = 1
        fp_full.append(sampled_fp)

    fp_full = torch.stack(fp_full)

    save_path = f"dataset/{dataset_name}/fp.pt"
    torch.save(fp_full, save_path)

def get_emb_sim(dataset_name, entity_type):
    dir_path = f"dataset/{dataset_name}"
    
    load_path = f"{dir_path}/{entity_type}_name_emb.npy"
    emb = np.load(load_path)

    dist = cdist(emb, emb, "cosine")
    sim = 2 - dist
    sim = np.clip(sim, 0, 2)
    sim = sim / 2

    sim = (sim - sim.min()) / (sim.max() - sim.min())
    
    sim = torch.from_numpy(sim).float()
    
    save_path = f"{dir_path}/{entity_type}_sim.pt"
    torch.save(sim, save_path)

def save_json_and_zip(data_all, dataset_name, mode, num_seeds, num_folds):
    dir_path = f"dataset/{dataset_name}"
    suffix = f"{num_seeds}_{num_folds}"
    
    save_path = f"{dir_path}/split_{mode}_{suffix}.json"
    
    with open(save_path, "w") as f:
        json.dump(data_all, f)
    
    save_path = f"{dir_path}/split_{mode}_{suffix}.zip"
    
    with zipfile.ZipFile(
        save_path, "w", zipfile.ZIP_BZIP2, compresslevel=9) as f:
        
        filename = f"{dir_path}/split_{mode}_{suffix}.json"
        arcname = f"split_{mode}_{suffix}.json"
        f.write(filename, arcname)

def split_train_test(dataset_name, num_seeds, num_folds):    
    if dataset_name != "repodb":
        load_path = f"dataset/{dataset_name}/adj_full.npz"
        adj_full = sparse.load_npz(load_path).toarray()
        num_drugs, num_diseases = adj_full.shape
        
        pos_pairs = adj_full.nonzero()
        neg_pairs = (1 - adj_full).nonzero()
        
        pos_pairs = np.stack(pos_pairs).T
        neg_pairs = np.stack(neg_pairs).T
    else:
        load_path = "dataset/repodb/pairs.json"
        with open(load_path, "r") as f:
            pairs = json.load(f)
        
        load_path = "dataset/repodb/drug_dict.json"
        with open(load_path, "r") as f:
            num_drugs = len(json.load(f))
        
        load_path = "dataset/repodb/disease_dict.json"
        with open(load_path, "r") as f:
            num_diseases = len(json.load(f))
        
        pos_pairs = list(filter(lambda x: x[-1] == 0, pairs))
        neg_pairs = list(filter(lambda x: x[-1] != 0, pairs))
        
        pos_pairs = [(cur[0], cur[1]) for cur in pos_pairs]
        neg_pairs = [(cur[0], cur[1]) for cur in neg_pairs]

        pos_pairs = np.array(pos_pairs)
        neg_pairs = np.array(neg_pairs)
    
    num_pos = pos_pairs.shape[0]
    num_neg = neg_pairs.shape[0]
    print(f"Pos: {num_pos}, Neg: {num_neg}")
    
    pos_dummy = np.zeros((num_pos, ))
    neg_dummy = np.zeros((num_neg, ))

    train_data_all = []
    test_data_all = []
    
    for seed in range(num_seeds):
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        
        kfold_iter_pos = kfold.split(pos_dummy)
        kfold_iter_neg = kfold.split(neg_dummy)
        
        train_data = []
        test_data = []
        
        for fold in range(num_folds):
            train_pos_ind, test_pos_ind = next(kfold_iter_pos)
            train_neg_ind, test_neg_ind = next(kfold_iter_neg)

            train_pos = pos_pairs[train_pos_ind].tolist()
            train_neg = neg_pairs[train_neg_ind].tolist()

            test_pos = pos_pairs[test_pos_ind].tolist()
            test_neg = neg_pairs[test_neg_ind].tolist()
            
            print(
                f"Seed: {seed}, Fold: {fold}, "
                f"Train Pos: {len(train_pos)}, Train Neg: {len(train_neg)}, "
                f"Test Pos: {len(test_pos)}, Test Neg: {len(test_neg)}"
            )
            
            if dataset_name == "repodb":
                cur_train_data = {
                    "train_pos" : train_pos,
                    "train_neg" : train_neg
                }
                
            else:
                cur_train_data = {
                    "train_pos" : train_pos,
                    "test" : test_pos + test_neg
                }
                
            cur_test_data = {
                "train_pos" : train_pos,
                "test_pos" : test_pos,
                "test_neg" : test_neg
            }
            
            train_data.append(cur_train_data)
            test_data.append(cur_test_data)
        
        train_data_all.append(train_data)
        test_data_all.append(test_data)
    
    save_json_and_zip(
        train_data_all, dataset_name, "train", num_seeds, num_folds
    )
    
    save_json_and_zip(
        test_data_all, dataset_name, "test", num_seeds, num_folds
    )

def split_train_valid(dataset_name, num_seeds, num_folds, valid_ratio):
    random.seed(0)
    
    dir_path = f"dataset/{dataset_name}"
    suffix = f"{num_seeds}_{num_folds}"

    load_path = f"{dir_path}/drug_dict.json"
    with open(load_path, "r") as f:
        num_drugs = len(json.load(f))

    load_path = f"{dir_path}/disease_dict.json"
    with open(load_path, "r") as f:
        num_diseases = len(json.load(f))
    
    load_path = f"{dir_path}/split_train_{suffix}.json"
    with open(load_path, "r") as f:
        data_all = json.load(f)

    train_valid_data_all = []
    
    for seed in range(num_seeds):
        train_valid_data = []
        
        for fold in range(num_folds):
            cur_data = data_all[seed][fold]
            
            train_valid_pos = cur_data["train_pos"]
            
            if "train_neg" in cur_data:
                train_valid_neg = cur_data["train_neg"]
            else:
                adj = np.ones((num_drugs, num_diseases))
                
                for split_data in cur_data.values():
                    drugs, diseases = map(list, zip(*split_data))
                    adj[drugs, diseases] = 0
                
                train_valid_neg = np.stack(adj.nonzero()).T.tolist()
            
            train_valid_pos_num = len(train_valid_pos)
            train_valid_neg_num = len(train_valid_neg)
            
            valid_pos_num = int(valid_ratio * train_valid_pos_num)
            valid_neg_num = int(valid_ratio * train_valid_neg_num)
            
            random.shuffle(train_valid_pos)
            random.shuffle(train_valid_neg)

            valid_pos = train_valid_pos[ : valid_pos_num]
            valid_neg = train_valid_neg[ : valid_neg_num]

            train_pos = train_valid_pos[valid_pos_num : ]
            train_neg = train_valid_neg[valid_neg_num : ]
            
            print(
                f"Seed: {seed}, Fold: {fold}, "
                f"Train Pos: {len(train_pos)}, Train Neg: {len(train_neg)}, "
                f"Valid Pos: {len(valid_pos)}, Valid Neg: {len(valid_neg)}"
            )
            
            if dataset_name == "repodb":
                cur_train_valid_data = {
                    "train_pos" : train_pos,
                    "train_neg" : train_neg,
                    "valid_pos" : valid_pos,
                    "valid_neg" : valid_neg
                }
                
            else:
                cur_train_valid_data = {
                    "train_pos" : train_pos,
                    "valid_pos" : valid_pos,
                    "valid_neg" : valid_neg
                }
            
            train_valid_data.append(cur_train_valid_data)
        
        train_valid_data_all.append(train_valid_data)
    
    save_json_and_zip(
        train_valid_data_all, dataset_name, "train_valid", num_seeds, num_folds
    )

def split_train_test_new_diseases(dataset_name, num_folds=5):
    # only for Fdataset, Cdataset, lrssl
    
    dir_path = f"dataset/{dataset_name}"
    load_path = f"{dir_path}/adj_full.npz"
    
    adj_full = sparse.load_npz(load_path).toarray()
    
    disease_degs = adj_full.sum(axis=0)
    new_diseases = disease_degs.argsort()[-num_folds : ].tolist()[::-1]
    
    save_path = f"{dir_path}/new_diseases_{num_folds}.json"
    with open(save_path, "w") as f:
        json.dump(new_diseases, f, indent=2)

    train_data = []
    test_data = []
    
    for i, disease_idx in enumerate(new_diseases):
        adj_pos = adj_full.copy()
        adj_pos[ : , disease_idx] = 0
        train_pos = np.stack(adj_pos.nonzero()).T.tolist()
        
        adj_neg = (1 - adj_full)
        adj_neg[ : , disease_idx] = 0
        train_neg = np.stack(adj_neg.nonzero()).T.tolist()
        
        pos_drug_mask = adj_full[ : , disease_idx]
        neg_drug_mask = 1 - pos_drug_mask
        
        pos_drugs = pos_drug_mask.nonzero()[0].tolist()
        neg_drugs = neg_drug_mask.nonzero()[0].tolist()
        
        test_pos = [(cur, disease_idx) for cur in pos_drugs]
        test_neg = [(cur, disease_idx) for cur in neg_drugs]

        print(
            f"{i}, Disease Idx: {disease_idx}, "
            f"Train Pos: {len(train_pos)}, Train Neg: {len(train_neg)}, "
            f"Test Pos: {len(test_pos)}, Test Neg: {len(test_neg)}"
        )
        
        cur_train_data = {
            "train_pos" : train_pos,
            "train_neg" : train_neg,
            "test" : test_pos + test_neg
        }
        
        cur_test_data = {
            "train_pos" : train_pos,
            "test_pos" : test_pos,
            "test_neg" : test_neg
        }
        
        train_data.append(cur_train_data)
        test_data.append(cur_test_data)

    train_data_all = [train_data]
    test_data_all = [test_data]
    
    save_json_and_zip(
        train_data_all, dataset_name, "train", 1, num_folds
    )
    
    save_json_and_zip(
        test_data_all, dataset_name, "test", 1, num_folds
    )

if __name__ == "__main__":
    #process_dataset("Fdataset")
    #process_dataset("Cdataset")
    #process_lrssl()
    #process_repodb()

    #get_rdkfp("Fdataset")
    #get_rdkfp("Cdataset")
    #get_rdkfp("lrssl")
    #get_rdkfp("repodb")
    #impute_fp("repodb")

    #get_emb_sim("repodb", "drug")
    #get_emb_sim("repodb", "disease")

    #split_train_test("Fdataset", 10, 10)
    #split_train_test("Cdataset", 10, 10)
    #split_train_test("lrssl", 10, 10)
    #split_train_test("repodb", 10, 10)

    #split_train_valid("Fdataset", 10, 10, 0.1)
    #split_train_valid("Cdataset", 10, 10, 0.1)
    #split_train_valid("lrssl", 10, 10, 0.1)
    #split_train_valid("repodb", 10, 10, 0.1)

    #split_train_test_new_diseases("Fdataset", 5)
    #split_train_valid("Fdataset", 1, 5, 0.1)
