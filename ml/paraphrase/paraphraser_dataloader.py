import os, json
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import numpy as np
import pandas as pd

def load_dataset(data_dir, tokenizer, mode, delimiter = "<|endoftext|>", prefix = "paraphrase: ", n_proc = 16):
    all_input_ids = None
    all_input_attention_mask = None
    all_output_ids = None
    filename = os.path.join(data_dir, f"{mode}_processed.txt")
    with open(filename,'r') as fob:
        data = fob.readlines()
    chunk_size = len(data)//n_proc + 1
    for chunk in tqdm(np.array_split(data, chunk_size)):
        chunk = [x.split(delimiter) for x in chunk]
        inputs = tokenizer([prefix+x[0] for x in chunk], padding="max_length", return_tensors='pt', truncation = True)
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

        
def load_dataset_style(data_dir, tokenizer, mode, task, delimiter = "<|endoftext|>", prefix = "paraphrase: ", n_proc = 16):
    all_input_ids = None
    all_attention_mask = None
    config_file = os.path.join(data_dir, task, "config.json")
    config = json.load(open(config_file))
    filename = os.path.join(data_dir, task, config['input_files'][mode])
    data = pd.read_csv(filename, header=None)
    chunk_size = len(data)//n_proc + 1
    for chunk in tqdm(np.array_split(data, chunk_size)):
        tokenized = tokenizer([prefix+x for x in list(chunk[0])], padding='max_length', return_tensors='pt', truncation = True)
        if all_input_ids is None:
            all_input_ids, all_attention_mask = tokenized['input_ids'], tokenized['attention_mask']
        else:
            all_input_ids = torch.cat((all_input_ids, tokenized['input_ids']),0)
            all_attention_mask = torch.cat((all_attention_mask, tokenized['attention_mask']), 0)
    dataset = TensorDataset(all_input_ids, all_attention_mask)
    return dataset


def load_dataset_psuedo(data_dir, tokenizer, mode, tasks, n_proc = 16):
    all_input_ids = None
    all_attention_mask = None
    all_output_ids = None
    for t, task in enumerate(tasks) :
        filename = os.path.join(data_dir, task, mode,'.csv')    
        data = pd.read_csv(filename, header=None)
        chunk_size = len(data)//n_proc + 1
        for chunk in tqdm(np.array_split(data, chunk_size)):
            temp = 'transfer: ' + chunk[0] + ' | input formality: '+chunk[1] + ' | output formality: '+chunk[3]
            tokenized = tokenizer(list(temp), padding='max_length', return_tensors='pt', truncation = True)
            output = tokenizer(list(chunk[2]), padding='max_length', return_tensors='pt', truncation = True)
            if all_input_ids is None:
                all_input_ids, all_attention_mask = tokenized['input_ids'], tokenized['attention_mask']
                all_output_ids = output['input_ids']
            else:
                all_input_ids = torch.cat((all_input_ids, tokenized['input_ids']),0)
                all_output_ids = torch.cat((all_output_ids, output['input_ids']),0)
                all_attention_mask = torch.cat((all_attention_mask, tokenized['attention_mask']), 0)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_output_ids)
    return dataset

def load_dataset_psuedo_diff(data_dir, tokenizer, mode, tasks, n_proc = 16):
    all_input_ids = None
    all_attention_mask = None
    all_output_ids = None
    for t, task in enumerate(tasks) :
        filename = os.path.join(data_dir, task, mode,'.csv')    
        data = pd.read_csv(filename, header=None)
        chunk_size = len(data)//n_proc + 1
        for chunk in tqdm(np.array_split(data, chunk_size)):
            temp = 'transfer: ' + chunk[0] + ' | input to output: '+chunk[2]
            tokenized = tokenizer(list(temp), padding='max_length', return_tensors='pt', truncation = True)
            output = tokenizer(list(chunk[1]), padding='max_length', return_tensors='pt', truncation = True)
            if all_input_ids is None:
                all_input_ids, all_attention_mask = tokenized['input_ids'], tokenized['attention_mask']
                all_output_ids = output['input_ids']
            else:
                all_input_ids = torch.cat((all_input_ids, tokenized['input_ids']),0)
                all_output_ids = torch.cat((all_output_ids, output['input_ids']),0)
                all_attention_mask = torch.cat((all_attention_mask, tokenized['attention_mask']), 0)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_output_ids)
    return dataset