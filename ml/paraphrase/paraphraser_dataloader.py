import os, json
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import numpy as np

def load_dataset(data_dir, tokenizer, mode, delimiter = "<|endoftext|>", n_proc = 16):
    all_input_ids = None
    all_input_attention_mask = None
    all_output_ids = None
    filename = os.path.join(data_dir, f"{mode}_processed.txt")
    with open(filename,'r') as fob:
        data = fob.readlines()
    chunk_size = len(data)//n_proc + 1
    for chunk in tqdm(np.array_split(data, chunk_size)):
        chunk = [x.split(delimiter) for x in chunk]
        inputs = tokenizer([x[0] for x in chunk], padding="max_length", return_tensors='pt', truncation = True)
        outputs = tokenizer([x[1] for x in chunk], padding="max_length", return_tensors='pt', truncation = True)
        if all_input_ids is None:
            all_input_ids, all_input_attention_mask = inputs['input_ids'], inputs['attention_mask']
            all_output_ids = outputs['input_ids']
        else:
            all_input_ids = torch.cat((all_input_ids, inputs['input_ids']),0)
            all_input_attention_mask = torch.cat((all_input_attention_mask, inputs['attention_mask']), 0)
            all_output_ids = torch.cat((all_output_ids, outputs['input_ids']),0)
        
        assert len(all_input_ids) == len(all_input_attention_mask)
        assert len(all_input_ids) == len(all_output_ids)
    dataset = TensorDataset(all_input_ids, all_input_attention_mask, all_output_ids)
    return dataset

        

