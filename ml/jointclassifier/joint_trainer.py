from typing import Any, List, Dict, Tuple, Optional, DefaultDict, Union
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch import cuda, nn, save, unsqueeze, sigmoid, stack, no_grad
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, logging
from tqdm.notebook import tqdm, trange
import os
import json
import numpy as np 
from sklearn.metrics import f1_score

logger = logging.get_logger()
logger.setLevel(logging.INFO)

class JointTrainer(object):
    def __init__(
        self,
        args: List[Any],
        model: Any,
        train_dataset: Optional[TensorDataset] = None,
        dev_dataset: Optional[TensorDataset] = None,
        idx_to_classes: Optional[Dict[str, Any]] = None
    ) -> None:
        self.args, self.model_args, self.data_args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.model = model
        # GPU or CPU
        self.device = ("cuda" if cuda.is_available() and not self.args.no_cuda else "cpu")
        self.model.to(self.device)
        self.tasks = self.data_args.task.split('+')
        self.idx_to_classes = idx_to_classes
        self.label_dims = label_dims = {task : 1 if len(list(idx_to_classes[task].keys())) == 2 else len(list(idx_to_classes[task].keys())) for task in idx_to_classes}
        


        
    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        writer = SummaryWriter()

        # Prepare optimizer and schedule (linear warmup and decay)
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
        #     'weight_decay': self.args.weight_decay},
        #     {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]

        optimizer_grouped_parameters = []
        for name, params in self.model.named_parameters():
            if self.model_args.freeze_encoder:
                if self.model_args.model_nick in name:
                    continue
            optimizer_grouped_parameters.append(params)
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"Num examples = {len(self.train_dataset)}")
        logger.info(f"Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"Total train batch size = {self.args.train_batch_size}")
        logger.info(f"Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"Total optimization steps = {t_total}")
        logger.info(f"Logging steps = {self.args.logging_steps}")
        logger.info(f"Save steps = {self.args.save_steps}")

        global_step = 0
        tr_loss = 0.0
        best_model_epoch = 0
        best_model_step = 0
        epoch_count = -1
        dev_f1_history, dev_step_history = [], []
        
        
        optimizer.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_count+=1
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = self.load_inputs_from_batch(batch)
                outputs = self.model(**inputs)
                loss_dict = outputs[0]
                loss = 0
                for task, value in loss_dict.items():
                    loss += value

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                # CHECK batch order and losses
                # from pdb import set_trace; set_trace()

                tr_loss += loss.item()
                writer.add_scalar('Train/avg_loss', tr_loss / (global_step+1), global_step)
                writer.add_scalar('Train/loss', loss, global_step)
                epoch_iterator.set_description("step {}/{} loss={:.2f}".format(
                        step,
                        global_step,
                        tr_loss / (global_step+1)
                    ))

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        results = self.evaluate()

                        result_to_save = {'model':self.model_args.model_name_or_path,
                                    'global_step':global_step }

                        for k,v in results.items():
                            result_to_save[k] = v

                        # save model
                        dev_f1 =  result_to_save["f1_mean"]
                        dev_loss =  result_to_save["dev_loss"]
                        writer.add_scalar('Dev/f1', dev_f1, global_step)
                        writer.add_scalar('Dev/loss', dev_loss, global_step)
                        if global_step == self.args.logging_steps  or dev_f1 > max(dev_f1_history):
                            self.save_model()
                            best_model_epoch = epoch_count
                            best_model_step = global_step
                            logger.info(f"New best model saved at step {global_step}, epoch {epoch_count}: f1 = {dev_f1}")
                        else:
                            logger.info(f"Best model still at step {best_model_step}, epoch {best_model_epoch}")
                        
                        dev_f1_history += [dev_f1]
                        dev_step_history += [global_step]
                        result_to_save['best_f1_mean'] = max(dev_f1_history)
                        result_to_save['best_global_step'] = best_model_step
                        result_to_save['best_global_epoch'] = best_model_epoch
                        # save log
                        filename = f'logs/logs_train_joint_{self.model_args.model_nick}_{self.data_args.task}.jsonl'
                        if not os.path.exists(os.path.dirname(filename)):
                            os.makedirs(os.path.dirname(filename))
                        with open(filename,'a') as f:
                            f.writelines(json.dumps(result_to_save) + '\n')
        return global_step, tr_loss / global_step


    def evaluate(self):
        dev_sampler = RandomSampler(self.dev_dataset)
        dev_dataloader = DataLoader(self.dev_dataset, sampler=dev_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running Evaluation *****")
        logger.info(f"Num examples = {len(self.dev_dataset)}")
        logger.info(f"Total eval batch size = {self.args.eval_batch_size}")

        global_step = 0
        e_loss = 0.0
        epoch_iterator = tqdm(dev_dataloader, desc="Iteration")
        all_labels_dict = {task:np.array([]) for task in self.tasks}
        all_preds_dict = {task:np.array([]) for task in self.tasks}
        with no_grad(): 
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = self.load_inputs_from_batch(batch)
                loss_dict, logits_dict, labels_dict = self.model(**inputs)
                loss = 0
                for task in self.tasks:
                    if task in logits_dict:
                        if self.label_dims[task]==1:
                            all_preds_dict[task] = np.append(all_preds_dict[task], (sigmoid(logits_dict[task])>0.5).float().squeeze().detach().cpu().numpy(), axis = 0)
                        else:
                            all_preds_dict[task] = np.append(all_preds_dict[task], logits_dict[task].argmax(axis=1).detach().cpu().numpy(), axis = 0)
                        all_labels_dict[task] = np.append(all_labels_dict[task], labels_dict[task].detach().cpu().numpy(), axis=0)
                        loss += loss_dict[task]
                e_loss += loss.item()
                epoch_iterator.set_description("step {}/{} loss={:.2f}".format(
                        step,
                        global_step,
                        e_loss / (global_step+1)
                    ))
                global_step += 1

        results = self.compute_stats(all_preds_dict, all_labels_dict)
        results['dev_loss'] = e_loss / global_step
        return results

    def compute_stats(self, preds_dict, labels_dict):
        results = {}
        f1_mean = 0
        for task in self.tasks:
            f1 = f1_score(labels_dict[task], preds_dict[task])
            results[f'{task}_f1']  =  f1
            f1_mean += f1
        results['f1_mean'] = f1_mean / len(self.tasks)
        return results

    
    def save_model(self):
        # Save model checkpoint (Overwrite)
        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)

        # Save training arguments together with the trained model
        save(self.args, os.path.join(output_dir, 'training_args.bin'))
        logger.info(f"Saving model checkpoint to {output_dir}")


    def load_inputs_from_batch(self, batch):

        # inputs = {'input_ids': batch[0],
                  # 'attention_mask': batch[1],
                  # 'label_ids': batch[3]}
        label_ids = {}
        
        # [batch x task]
        if len(batch[3].shape) == 1:
            label_ids_tensors = unsqueeze(batch[3],-1)
        else:
            label_ids_tensors = batch[3]
        # [batch x length x task]
        for idx, task in enumerate(self.tasks):
            label_ids[task] = label_ids_tensors[:,idx]
            
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels_all' : label_ids,
                  'task_ids' : batch[4]
                }
        if 'distilbert' not in self.model_args.model_name_or_path:
            inputs['token_type_ids'] = batch[2]
        return inputs


    def predict_for_sentence(self, sentence, tokenizer, salience=False):
        '''
        Use the model to make a classification on a given sentence. 
        
        Arguments:
          sentence : str. The sentence to classify.
          tokenizer : HuggingFace transformers tokenizer.
          salience : bool (optional). Boolean flag for whether to return saliency maps
          
        Returns:
          predictions: dict. 
            Contains keys for each of the joint_trainer's tasks.
            For each task, there is another dict with keys 'class'
            for classification class label as a string, 'prob' for the probabilty
            score for this class as a string representation of a float, 
            and 'salience' as an array of string representations of floats for each
            the salience of each token from the input sentence.
        '''
        self.model.eval()
        tokenized_sentence = tokenizer(sentence, padding='max_length', return_tensors='pt', truncation = True)
        input_ids = tokenized_sentence['input_ids'].to(self.device)
        attention_mask = tokenized_sentence['attention_mask'].to(self.device)
        if 'distilbert' not in self.model_args.model_nick:
            token_type_ids = tokenized_sentence['token_type_ids'].to(self.device)
            if salience:
                logits_dict, hidden_states = self.model.predict(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
            else:
                logits_dict = self.model.predict(input_ids, attention_mask, token_type_ids)

        else:
            if salience:
                logits_dict, hidden_states = self.model.predict(input_ids, attention_mask, output_hidden_states=True)
            else:
                logits_dict = self.model.predict(input_ids, attention_mask)

        predictions = {}
        if salience:
            for task in self.tasks:
                preds = int((sigmoid(logits_dict[task])>0.5).float().squeeze().detach().cpu().numpy())
                predictions[task] = {"class" : self.idx_to_classes[task][str(preds)], "prob": str(sigmoid(logits_dict[task]).squeeze().detach().cpu().numpy()), 'salience' : {}}
                class_score = logits_dict[task][0]
                self.model.zero_grad()
                class_score.backward(retain_graph=True)
                grads = [h_state.grad for h_state in hidden_states]
                temp = stack(grads).abs().max(axis=0)[0].squeeze()[attention_mask.squeeze()>0][1:-1]
                temp = temp.mean(dim=1)
                temp = list((temp / temp.sum(dim=0)).detach().cpu().numpy())
                predictions[task]['salience'] = [str(x) for x in temp if x!=0]
            return predictions
        else:
            for task in self.tasks:
                preds = int((logits_dict[task]>0.5).float().squeeze().detach().cpu().numpy())
                predictions[task] = {"class" : self.idx_to_classes[task][str(preds)], "prob": str(sigmoid(logits_dict[task].squeeze()).detach().cpu().numpy())}
            return predictions
