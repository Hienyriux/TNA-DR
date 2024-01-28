# -*- coding: utf-8 -*-

import sys
import json
import random

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader, BatchSampler

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Dict, Pad
from paddlenlp.transformers import AutoModel, AutoTokenizer

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def read_dataset(dataset_name, entity_type):
    dir_path = f"dataset/{dataset_name}"
    load_path = f"{dir_path}/{entity_type}_id_to_name.json"
    
    with open(load_path, "r") as f:
        name_list = list(json.load(f).values())
    
    for name in name_list:
        yield name

def tokenize_text(text, tokenizer):
    return tokenizer(text=text)

def get_emb(dataset_name, entity_type, model, model_name, batch_size=8):
    data_set = load_dataset(
        read_dataset,
        dataset_name=dataset_name,
        entity_type=entity_type,
        lazy=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_set.map(tokenizer)
    
    data_set_len = len(data_set)
    
    batchify_fn = lambda samples, fn=Dict({
        "input_ids" : Pad(axis=0, pad_val=tokenizer.pad_token_id),
    }): fn(samples)
    
    data_sampler = BatchSampler(
        data_set,
        batch_size=batch_size,
        shuffle=False
    )
    
    data_loader = DataLoader(
        dataset=data_set,
        batch_sampler=data_sampler,
        collate_fn=batchify_fn
    )
    
    emb_list = []
    cur_num = 0
    
    model.eval()
    
    for i, batch in enumerate(data_loader()):
        input_ids = batch[0]
        
        with paddle.no_grad():            
            out, _ = model(input_ids)
        
        out = out.sum(axis=1)
        out = out.detach().cpu()
        
        emb_list.append(out)
        
        cur_num += out.shape[0]
        sys.stdout.write(f"\r{cur_num} / {data_set_len}")
        sys.stdout.flush()

    emb_list = paddle.concat(emb_list)
    emb_list = emb_list.numpy()
    
    save_path = f"dataset/{dataset_name}/{entity_type}_name_emb.npy"
    np.save(save_path, emb_list)
    
if __name__ == "__main__":
    set_all_seeds(0)
    
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    
    model = AutoModel.from_pretrained(model_name)
    
    get_emb("repodb", "disease", model, model_name, batch_size=8)
    get_emb("repodb", "drug", model, model_name, batch_size=8)
