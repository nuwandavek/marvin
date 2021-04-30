import sys, os
sys.path.append('../paraphrase/')
from paraphraser_args import ModelArguments, DataTrainingArguments, TrainingArguments
from paraphraser_dataloader import load_dataset_pseudo_diff, load_dataset_pseudo
from paraphraser_trainer import ParaphraserTrainer
from transformers import AutoTokenizer, AutoModelWithLMHead, HfArgumentParser

data_dir = "../data/pseudo"
task = "formality"
model_name = "t5-small"
model_nick = "t5_transfer_formality"
output_dir = "../models/"
epochs = "4"
train_batch_size = "16"
eval_batch_size = "16"
save_log_steps = "900"

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses([
    "--model_name_or_path",
    model_name,
    "--model_nick",
    model_nick,
    "--data_dir",
    data_dir,
    "--output_dir",
    os.path.join(output_dir, model_nick),
    "--cache_dir",
    os.path.join(output_dir,"cache"),
    "--overwrite_cache",
    "--per_device_train_batch_size",
    train_batch_size,
    "--per_device_eval_batch_size",
    eval_batch_size,
    "--max_seq_len",
    "64",
    "--gradient_accumulation_steps",
    "1",
    "--num_train_epochs",
    epochs,
    "--logging_steps",
    save_log_steps,
    "--save_steps",
    save_log_steps,
    "--data_parallel",
    "True",
    "--meta_task",
    "transfer",
    "--meta_task_type",
    "intra"
])


tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
model = AutoModelWithLMHead.from_pretrained(model_args.model_name_or_path)

if training_args.meta_task_type=='intra':
    train_dataset = load_dataset_pseudo(data_args.data_dir, tokenizer, mode="train", tasks = task.split('+'), n_proc=2048)
    dev_dataset = load_dataset_pseudo(data_args.data_dir, tokenizer, mode="dev",  tasks = task.split('+'), n_proc=2048)
elif training_args.meta_task_type=='diff':
    train_dataset = load_dataset_pseudo_diff(data_args.data_dir, tokenizer, mode="train", tasks = task.split('+'), n_proc=2048)
    dev_dataset = load_dataset_pseudo_diff(data_args.data_dir, tokenizer, mode="dev",  tasks = task.split('+'), n_proc=2048)



trainer = ParaphraserTrainer([training_args,model_args, data_args], model, tokenizer, train_dataset, dev_dataset)
trainer.train()