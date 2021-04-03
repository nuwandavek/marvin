from transformers import DistilBertPreTrainedModel, DistilBertModel
import torch
import torch.nn as nn


class JointSeqClassifier(DistilBertPreTrainedModel):
    '''
    A class that inherits from DistilBertForSequenceClassification, but extends the model to 
    have multiple classifiers at the end to perform joint classification over multiple tasks.
    '''
    def __init__(self, config, tasks, model_args, task_if_single, joint, intermediate_layer_dim = 100):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.tasks = tasks
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        # List of classifiers
        self.classifiers = {}
        self.intermediates = {}
        for task in tasks:
            self.intermediates[task] = nn.Linear(config.dim, intermediate_layer_dim)
            self.classifiers[task] = nn.Linear(intermediate_layer_dim, config.num_labels)
        
        self.classifier = nn.ModuleDict(self.classifiers)
        self.intermediates = nn.ModuleDict(self.intermediates)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.skip_preclassifier = model_args.skip_preclassifier
        self.task_if_single = task_if_single
        self.joint = joint
        self.init_weights()
        
    def forward(
        self,
        input_ids, 
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels_all=None,
        task_ids = None
    ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        
        task_ids (list of ints):
            Labels indexing which classification task the labels correspond to.
        """
                
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_state = distilbert_output['last_hidden_state']  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        if not self.skip_preclassifier:
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        
        logits_dict = {}
        selected_labels_dict = {}
        if self.joint:
            tasks = self.tasks
        else:
            tasks = [self.task_if_single]
        loss_dict = {task:0 for task in tasks}
        for t, task in enumerate(tasks):
            logits = self.intermediates[task](pooled_output)  # (bs, intermediate_layer_dim)
            logits = nn.ReLU()(logits)
            logits = self.classifier[task](logits)  # (bs, num_labels)
            logits = logits[task_ids==t]
            if len(logits)==0:
                continue
            labels = labels_all[task][task_ids==t]
            
            logits_dict[task] = logits
            selected_labels_dict[task] = labels
            if labels != None:
                if self.num_labels == 1:
                    loss_fct = nn.MSELoss()
                    loss_dict[task] += loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss_dict[task] += loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    
        return (loss_dict, logits_dict, selected_labels_dict)
        

    def predict(
        self,
        input_ids, 
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        hidden_state = distilbert_output['last_hidden_state']  # (1, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (1, dim)
        if not self.skip_preclassifier:
            pooled_output = self.pre_classifier(pooled_output)  # (1, dim)
            pooled_output = nn.ReLU()(pooled_output)  # (1, dim)
        pooled_output = self.dropout(pooled_output)  # (1, dim)
        
        logits_dict = {}
        for t, task in enumerate(self.tasks):
            logits = self.intermediates[task](pooled_output)  # (bs, intermediate_layer_dim)
            logits = nn.ReLU()(logits)
            logits_dict[task] = self.classifier[task](logits)  # (bs, num_labels)        
        if output_hidden_states:
            all_hidden_states = distilbert_output['hidden_states']
            for h_state in all_hidden_states:
                h_state.retain_grad()

            return logits_dict, all_hidden_states
        else:
            return logits_dict
        
    

    
