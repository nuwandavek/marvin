import os, json
import torch
from torch.utils.data import TensorDataset

def load_dataset(args, tokenizer, model_name, tasks, mode):

    all_input_ids = torch.Tensor([], dtype = torch.long)
    all_attention_mask = torch.Tensor([], dtype = torch.long)
    all_token_type_ids = torch.Tensor([], dtype = torch.long)
    all_labels = torch.Tensor([], dtype = torch.long)
    all_task_ids = torch.Tensor([], dtype = torch.long)
    for t, task in enumerate(tasks) :
        config_file = os.path.join(args.data_dir, task, "config.json")
        config = json.load(open(config_file))
            
        filename = config['input_file'][mode]
        with open(filename,'r') as fob:
            data = fob.readlines()
        task_labels = []
        for d in data:
            idx, sentence, label = d.split(',')
            tokenized = tokenizer(sentence, padding='max_length', return_tensors='pt')
            all_input_ids = torch.cat((all_input_ids, tokenized['input_ids']),0)
            all_attention_mask = torch.cat((all_attention_mask, tokenized['attention_mask']), 0)
            if 'distilbert' not in model_name:
                all_token_type_ids = torch.cat((all_token_type_ids, tokenized['token_type_ids']), 0)
            all_task_ids = torch.cat((all_token_type_ids, t), 0)
            task_labels += [label]
        task_labels = torch.tensor(task_labels, dtype=torch.long)
        task_labels_full = torch.zeros((len(data),len(tasks)), dtype = torch.long)
        task_labels_full[:,t] = task_labels
        all_labels = torch.cat((all_labels, task_labels_full), 0)

        assert len(all_input_ids) == len(all_attention_mask)
        if 'distilbert' not in model_name:
            assert len(all_input_ids) == len(all_token_type_ids)
        assert len(all_input_ids) == len(all_labels)
        assert len(all_labels[0]) == len(tasks)
        assert len(all_input_ids) == len(all_task_ids)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_task_ids)
    return dataset

        

