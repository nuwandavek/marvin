from typing import Any, List, Dict, Tuple, Optional, DefaultDict, Union
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch import cuda, nn, save, unsqueeze, sigmoid, stack
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, logging
from tqdm.notebook import tqdm, trange
import os
import json
import numpy as np 
from datasets import load_metric

logger = logging.get_logger()
logger.setLevel(logging.INFO)

class ParaphraserTrainer(object):
    def __init__(
        self,
        args: List[Any],
        model: Any,
        tokenizer : Any,
        train_dataset: Optional[TensorDataset] = None,
        dev_dataset: Optional[TensorDataset] = None,
    ) -> None:
        self.args, self.model_args, self.data_args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.model = model
        # GPU or CPU
        self.device = ("cuda" if cuda.is_available() and not self.args.no_cuda else "cpu")
        self.model.to(self.device)
        self.tokenizer = tokenizer
        
    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        writer = SummaryWriter()

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"Num examples = {self.train_dataset}")
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
        dev_score_history, dev_step_history = [], []
        
        
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_count+=1
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = self.load_inputs_from_batch(batch)
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                
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
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        results = self.evaluate()

                        result_to_save = {'model':self.model_args.model_name_or_path,
                                    'global_step':global_step }

                        for k,v in results.items():
                            result_to_save[k] = v

                        # save model
                        dev_score =  result_to_save["bleu"]
                        dev_loss =  result_to_save["dev_loss"]
                        writer.add_scalar('Dev/score', dev_score, global_step)
                        writer.add_scalar('Dev/loss', dev_loss, global_step)
                        if global_step == self.args.logging_steps  or dev_score > max(dev_score_history):
                            self.save_model()
                            best_model_epoch = epoch_count
                            best_model_step = global_step
                            logger.info(f"New best model saved at step {global_step}, epoch {epoch_count}: score = {dev_score}")
                        else:
                            logger.info(f"Best model still at step {best_model_step}, epoch {best_model_epoch}")
                        
                        dev_score_history += [dev_score]
                        dev_step_history += [global_step]
                        result_to_save['best_score_mean'] = max(dev_score_history)
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
        logger.info(f"Num examples = {self.dev_dataset}")
        logger.info(f"Total eval batch size = {self.args.eval_batch_size}")

        global_step = 0
        e_loss = 0.0
        
        self.model.eval()
        predicted = []
        labels = []
        epoch_iterator = tqdm(dev_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
            inputs = self.load_inputs_from_batch(batch)
            outputs = self.model(**inputs)
            generated_outputs = self.model.generate(inputs['input_ids'], inputs['attention_mask'])
            predicted += self.tokenizer.batch_decode(generated_outputs.detach().cpu.numpy(), skip_special_tokens=True)
            labels += self.tokenizer.batch_decode(inputs['labels'].detach().cpu.numpy(), skip_special_tokens=True)
            loss = outputs.loss
            e_loss += loss.item()
            epoch_iterator.set_description("step {}/{} loss={:.2f}".format(
                    step,
                    global_step,
                    e_loss / (global_step+1)
                ))
            global_step += 1

        results = self.compute_stats(predicted, labels)
        results['dev_loss'] = e_loss / global_step
        return results

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels
    
    def compute_stats(self, preds, labels):
        metric = load_metric("sacrebleu")
        preds, labels = self.postprocess_text(preds, labels)
        result = metric.compute(predictions=preds, references=labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    
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
        
        # [batch x length x task]
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
               }
        return inputs


    def predict_for_sentence(self, sentence):
        self.model.eval()
        tokenized_sentence = self.tokenizer(sentence, padding='max_length', return_tensors='pt', truncation = True)
        input_ids = tokenized_sentence['input_ids'].to(self.device)
        attention_mask = tokenized_sentence['attention_mask'].to(self.device)
        prediction = self.model.generate(input_ids, attention_mask)
        prediction = self.tokenizer.decode(prediction.detach().cpu.numpy()[0], skip_special_tokens=True).strip()
        return prediction
        