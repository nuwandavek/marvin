import os, json
import torch
from torch.utils.data import TensorDataset
import pandas as pd
from tqdm import tqdm

def load_dataset(data_dir, tokenizer, model_name, tasks, mode):

    all_input_ids = None
    all_attention_mask = None
    all_token_type_ids = None
    all_labels = None
    all_task_ids = None
    for t, task in enumerate(tasks) :
        config_file = os.path.join(data_dir, task, "config.json")
        config = json.load(open(config_file))
            
        filename = os.path.join(data_dir, task, config['input_files'][mode])
        data = pd.read_csv(filename, header=None)
        task_labels = []
        for r, row in tqdm(data.iterrows()):
            sentence, label = row
            tokenized = tokenizer(sentence, padding='max_length', return_tensors='pt', truncation = True)
            if all_input_ids is None:
                all_input_ids, all_attention_mask = tokenized['input_ids'], tokenized['attention_mask']
                if 'distilbert' not in model_name:
                    all_token_type_ids = tokenized['token_type_ids']
                all_task_ids = torch.ones(len(tokenized['input_ids'])) * t
            else:
                all_input_ids = torch.cat((all_input_ids, tokenized['input_ids']),0)
                all_attention_mask = torch.cat((all_attention_mask, tokenized['attention_mask']), 0)
                if 'distilbert' not in model_name:
                    all_token_type_ids = torch.cat((all_token_type_ids, tokenized['token_type_ids']), 0)
                all_task_ids = torch.cat((all_task_ids, torch.ones(len(tokenized['input_ids'])) * t), 0)
            task_labels += [label for i in range(len(tokenized['input_ids']))]
        task_labels = torch.tensor(task_labels, dtype=torch.long)
        task_labels_full = torch.zeros((len(data),len(tasks)), dtype = torch.long)
        task_labels_full[:,t] = task_labels
        if all_labels is None :
            all_labels = task_labels_full
        else:
            all_labels = torch.cat((all_labels, task_labels_full), 0)

        assert len(all_input_ids) == len(all_attention_mask)
        if 'distilbert' not in model_name:
            assert len(all_input_ids) == len(all_token_type_ids)
        assert len(all_input_ids) == len(all_labels)
        assert len(all_labels[0]) == len(tasks)
        assert len(all_input_ids) == len(all_task_ids)

        print(all_input_ids.shape, all_attention_mask.shape, all_labels.shape, all_task_ids.shape)

    if 'distilbert' in model_name:
        all_token_type_ids = torch.zeros_like(all_input_ids)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_task_ids)
    return dataset

        

