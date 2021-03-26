from transformers import DistilBertPreTrainedModel, DistilBertModel
import torch
import torch.nn as nn


class JointSeqClassifier(DistilBertPreTrainedModel):
    '''
    A class that inherits from DistilBertForSequenceClassification, but extends the model to 
    have multiple classifiers at the end to perform joint classification over multiple tasks.
    '''
    def __init__(self, config, tasks, model_args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.tasks = tasks
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        # List of classifiers
        self.classifiers = {}
        for task in tasks:
            self.classifiers[task] = nn.Linear(config.dim, config.num_labels)
        self.classifier = nn.ModuleDict(self.classifiers)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.skip_preclassifier = model_args.skip_preclassifier

        self.init_weights()
        
    def forward(
        self,
        input_ids, 
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels_all=None,
        output_attentions=None,
        output_hidden_states=None,
        task_ids = None
    ):
        r"""
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        if not self.skip_preclassifier:
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        
        logits_dict = {}
        selected_labels_dict = {}
        loss_dict = {task:0 for task in self.tasks}
        for t, task in enumerate(self.tasks):
            logits = self.classifier[task](pooled_output)  # (bs, num_labels)
            logits = logits[task_ids==t]
            if len(logits)==0:
                continue
            labels = labels_all[:,t][task_ids==t]
            
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