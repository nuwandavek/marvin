from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    TrainingArguments
  )



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_nick: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model Nickname"
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    freeze_encoder: Optional[bool] = field(
        default=False, metadata={"help" : "Freeze the encoder"}
    )
    skip_preclassifier: Optional[bool] = field(
        default=False, metadata={"help" : "Skip the preclassifier layer"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    ignore_index: Optional[int] = field(
        default=0,
        metadata={
            "help": "Specifies a target value that is ignored and does not contribute to the input gradient"},
    )
    
    dropout_rate: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Dropout for fully-connected layers"},
    )
    train_jointly: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Dropout for fully-connected layers"},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
      default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    task: Optional[str] = field(
      default=None,
      metadata={"help": "The name of the task to train"},
    )
    data_dir: Optional[str] = field(
      default='./data',
      metadata={"help": "The input data dir"},
    )
    max_seq_len: Optional[int] = field(
      default=50,
      metadata={"help": "TBW"},
    )
    result_dir: Optional[str] = field(
      default='./',
      metadata={"help": "The result dir"},
    )
    prediction_dir: Optional[str] = field(
      default='./',
      metadata={"help": "The prediction dir"},
    )



