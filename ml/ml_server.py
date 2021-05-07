# import flask related modules
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

# basic imports
import json
import sys

# Pytorch imports
import torch
from torchtext.data.utils import get_tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelWithLMHead

# Import our ML code
from attention.selfAttention import SelfAttention 
import attention.attention_utils
import bert.bert_utils
from attention.utils import load_pytorch_model, load_vocab_dict
from config.model_config import get_configs

# Joint Model imports
from jointclassifier.joint_args import ModelArguments, DataTrainingArguments, TrainingArguments
from jointclassifier.joint_dataloader import load_dataset
from jointclassifier.joint_trainer import JointTrainer
from jointclassifier.single_trainer import SingleTrainer
from jointclassifier.joint_model_v1 import JointSeqClassifier

from transformers import HfArgumentParser, AutoConfig, AutoTokenizer
import os


app = Flask(__name__)
CORS(app)

MODEL_PATHS = {
    'micro-formality' : {
        "classifier": "./models/distilbert_uncased_2/formality+emo/joint",
        "classifier_name": "distilbert-base-uncased",
        "classifier_nick": "distilbert_uncased_2",
        "classifier_task" : "formality+emo",
        "idx_to_classes" : {
            'formality': {'0': 'informal', '1': 'formal'},
            'emo': {'0': 'sad', '1': 'happy'}
        },
        "label_dims" : {'formality': 1, 'emo': 1},
        "transfer" : "./models/t5_transfer_formality_3",
        "transfer_name" : "t5-small",
        "transfer_nick" : "t5_transfer_formality_3",
    },
    'micro-joint' : {
        "classifier": "./models/distilbert_uncased_2/formality+emo/joint",
        "classifier_name": "distilbert-base-uncased",
        "classifier_nick": "distilbert_uncased_2",
        "classifier_task" : "formality+emo",
        "idx_to_classes" : {
            'formality': {'0': 'informal', '1': 'formal'},
            'emo': {'0': 'sad', '1': 'happy'}
        },
        "label_dims" : {'formality': 1, 'emo': 1},
        "transfer" : "./models/t5_transfer_formality_joint",
        "transfer_name" : "t5-small",
        "transfer_nick" : "t5_transfer_formality_joint",
    },
    'macro-shakespeare' : {
        "classifier": "./models/distilbert_uncased_2/shakespeare/joint",
        "classifier_name": "distilbert-base-uncased",
        "classifier_nick": "distilbert_uncased_2",
        "classifier_task" : "shakespeare",
        "idx_to_classes" : {
            'shakespeare': {'0': 'noshakespeare', '1': 'shakespeare'}
        },
        "label_dims" : {'shakespeare': 1},
        "transfer" : "./models/t5_transfer_shakespeare",
        "transfer_name" : "t5-small",
        "transfer_nick" : "t5_transfer_shakespeare",
    },
    'macro-binary' : {
        "transfer_name" : "t5-small",
        "transfer_shake" : "./models/t5_transfer_shakespeare_binary",
        "transfer_nick_shake" : "t5_transfer_shakespeare_binary",
        "transfer_abs" : "./models/t5_transfer_abstract_binary",
        "transfer_nick_abs" : "t5_transfer_abstract_binary",
        "transfer_wiki" : "./models/t5_transfer_wiki_binary",
        "transfer_nick_wiki" : "t5_transfer_wiki_binary",
    }
}


def load_models(mode):
    global classifier_tokenizer, classifier_trainer, classifier_model, transfer_model, transfer_tokenizer, transfer_model_shake, transfer_model_abs, transfer_model_wiki
    if mode in ['micro-formality','micro-joint','macro-shakespeare']:
        transfer_model_shake = None
        transfer_model_abs = None
        transfer_model_wiki = None
        
        mode_paths = MODEL_PATHS[mode]
        model_args = ModelArguments(
            model_name_or_path=mode_paths['classifier_name'],
            model_nick=mode_paths['classifier_nick'],
            cache_dir="./models/cache"
        )

        data_args = DataTrainingArguments(
            max_seq_len=64,
            task=mode_paths['classifier_task']
        )

        training_args = TrainingArguments(
            output_dir = mode_paths['classifier'],
            train_jointly= True
        )
        idx_to_classes = mode_paths['idx_to_classes']

        label_dims = mode_paths['label_dims']

        classifier_model = JointSeqClassifier.from_pretrained(
            training_args.output_dir,
            tasks=data_args.task.split('+'),
            model_args=model_args,
            task_if_single=None, 
            joint = training_args.train_jointly,
            label_dims=label_dims
        )
        classifier_trainer = JointTrainer(
            [training_args,model_args, data_args], 
            classifier_model, idx_to_classes = idx_to_classes
        )
        classifier_tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, 
            cache_dir=model_args.cache_dir,
            model_max_length = data_args.max_seq_len
        )

        transfer_tokenizer = AutoTokenizer.from_pretrained(mode_paths['transfer_name'])
        transfer_model = AutoModelWithLMHead.from_pretrained(mode_paths['transfer'])   
    elif mode in ['macro-binary']:
        classifier_model = None
        transfer_model = None
        mode_paths = MODEL_PATHS[mode]

        transfer_tokenizer = AutoTokenizer.from_pretrained(mode_paths['transfer_name'])
        transfer_model_shake = AutoModelWithLMHead.from_pretrained(mode_paths['transfer_shake'])
        transfer_model_abs = AutoModelWithLMHead.from_pretrained(mode_paths['transfer_abs'])
        transfer_model_wiki = AutoModelWithLMHead.from_pretrained(mode_paths['transfer_wiki'])
        

def bucket_match(bucket1, bucket2):
    if bucket2 == 'high':
        if bucket1 in ['mid','high']:
            return True
        else:
            return False
    elif bucket2 == 'low':
        if bucket1 in ['mid','low']:
            return True
        else:
            return False
    elif bucket2 == 'mid':
        if bucket1 in ['mid','low']:
            return True
        else:
            return False
    
    

def get_buckets(prob, classname):
    if classname=='formality':
        if prob<0.2:
            return 'low'
        elif prob>0.9:
            return 'high'
        else:
            return 'mid'
    elif classname=='emo':
        if prob<0.25:
            return 'low'
        elif prob>0.9:
            return 'high'
        else:
            return 'mid'
    if classname=='shakespeare':
        if prob<0.1:
            return 'low'
        elif prob>0.9:
            return 'high'
        else:
            return 'mid'


@app.route("/hello")
def hello():
    res = {
        "world": 42,
        "app": "ml"
    }
    return res


@app.route("/swap_models", methods=['POST'])
def swap_models():
    mode = request.args.get('mode', type = str)
    print(mode)
    try:
        load_models(mode)
    except Exception as e:
       print(e)
       return {'message' : 'Models Swap Failure! :('}, 500
 
    return {'message' : 'Models Swap Success! :)'}, 200


@app.route('/classification', methods = ['GET'])
def get_joint_classify_and_salience():
    '''
    Inputs:
    Input is assumed to be json of the form 
      {text: "some text"}.
  
      Results:
      Run ML classification model on text. 
      
      Returns:
    classification and attention weights
      '''
    # Get text input from request
    text = request.args.get('text', type = str)
    text = text.strip()
    lower = text.lower()
    tokens = []
    sentence_seen = 0

    joint_tokens = classifier_tokenizer.convert_ids_to_tokens(classifier_tokenizer.encode(lower))[1:-1]
    for token in joint_tokens:
        if len(token) > 2:
          if token[:2] == '##':
            token = token[2:]
            print(token)
        occ = lower[sentence_seen:].find(token)
        start = occ + sentence_seen
        end = start + len(token)
        adj_len = len(token)
        sentence_seen = sentence_seen + adj_len + occ
        tokens.append({'text' : text[start:end], 'start' : start, 'end' : end})
    
    
    res = classifier_trainer.predict_for_sentence(lower, classifier_tokenizer, salience=True)
    res['tokens'] = tokens
    # print(f"JointClassify RES\n{res}")
    return res, 200

@app.route('/transfer', methods = ['GET'])
def get_transfer():
    # Get text input from request
    text = request.args.get('text', type = str)
    mode = request.args.get('mode', type = str)
    controls = request.args.get('controls', type = str)
    text = text.strip()
    # lower = text.lower()
    lower = text
    controls = json.loads(controls)

    print(controls)

    if mode=="micro-formality":
        classifier_output = classifier_trainer.predict_for_sentence(lower, classifier_tokenizer, salience=False)
        input_bucket = get_buckets(float(classifier_output['formality']['prob']), 'formality')
        output_bucket = ['low', 'mid', 'high'][int(controls['formality'])]
        transfer_input = "transfer: "+lower+' | input: '+input_bucket + ' | output: '+output_bucket

        t = transfer_tokenizer(transfer_input, return_tensors='pt')
        gen = transfer_model.generate(input_ids= t.input_ids, attention_mask = t.attention_mask, max_length=70, 
                                            num_beams=15,
                                            #    early_stopping=True,
                                            encoder_no_repeat_ngram_size=5,
                                            no_repeat_ngram_size=3,
                                            num_beam_groups=5,
                                            diversity_penalty=0.5,
                                            # num_return_sequences=int(controls['suggestions'])
                                            num_return_sequences=10
                                            )
        transfers = transfer_tokenizer.batch_decode(gen, skip_special_tokens=True)

        res = {
            'input' : {
                'text' : text,
                'probs' : {
                    'formality' : classifier_output['formality']['prob']
                },
            },
            "goal" : f"Formality : {output_bucket}",
        }
        suggestions = []
        for transfer in transfers:
            cls_opt = classifier_trainer.predict_for_sentence(transfer, classifier_tokenizer, salience=False)
            temp = {
                'text' : transfer,
                'probs' : {
                    'formality' : cls_opt['formality']['prob']
                }
            }
            suggestions.append(temp)

        suggestions = [x for x in suggestions if bucket_match(get_buckets(float(x['probs']['formality']),'formality'),output_bucket)]
        if len(suggestions)>0:
            suggestions.sort(key=lambda x: float(x['probs']['formality']), reverse=True)
            res['suggestions'] = suggestions[:int(controls['suggestions'])]
        else:
            res['suggestions'] = []
        
    elif mode=="macro-shakespeare":
        classifier_output = classifier_trainer.predict_for_sentence(lower, classifier_tokenizer, salience=False)
        input_bucket = get_buckets(float(classifier_output['shakespeare']['prob']), 'shakespeare')
        output_bucket = ['low', 'mid', 'high'][int(controls['shakespeare'])]
        transfer_input = "transfer: "+lower+' | input: '+input_bucket + ' | output: '+output_bucket

        t = transfer_tokenizer(transfer_input, return_tensors='pt')
        gen = transfer_model.generate(input_ids= t.input_ids, attention_mask = t.attention_mask, max_length=70, 
                                            num_beams=12,
                                            #    early_stopping=True,
                                            encoder_no_repeat_ngram_size=5,
                                            no_repeat_ngram_size=3,
                                            num_beam_groups=3,
                                            diversity_penalty=0.5,
                                            num_return_sequences=int(controls['suggestions'])
                                            )
        transfers = transfer_tokenizer.batch_decode(gen, skip_special_tokens=True)

        res = {
            'input' : {
                'text' : text,
                'probs' : {
                    'shakespeare' : classifier_output['shakespeare']['prob']
                },
            },
            "goal" : f"Shakespeare : {output_bucket}",
            "suggestions":[]
        }
        suggestions = []
        for transfer in transfers:
            cls_opt = classifier_trainer.predict_for_sentence(transfer, classifier_tokenizer, salience=False)
            temp = {
                'text' : transfer,
                'probs' : {
                    'shakespeare' : cls_opt['shakespeare']['prob']
                }
            }
            suggestions.append(temp)

        suggestions = [x for x in suggestions if bucket_match(get_buckets(float(x['probs']['shakespeare']),'shakespeare'),output_bucket)]
        if len(suggestions)>0:
            suggestions.sort(key=lambda x: float(x['probs']['shakespeare']), reverse=True)
            res['suggestions'] = suggestions[:int(controls['suggestions'])]
        else:
            res['suggestions'] = []
        
    elif mode=="micro-joint":
        classifier_output = classifier_trainer.predict_for_sentence(lower, classifier_tokenizer, salience=False)
        input_bucket_f = get_buckets(float(classifier_output['formality']['prob']), 'formality')
        input_bucket_e = get_buckets(float(classifier_output['emo']['prob']), 'emo')
        output_bucket_f = ['low', 'mid', 'high'][int(controls['formality'])]
        output_bucket_e = ['low', 'mid', 'high'][int(controls['emo'])]
        transfer_input = 'transfer: ' + lower + ' | input formality: '+input_bucket_f + ' | input emotion: '+input_bucket_e +' | output formality: '+output_bucket_f +' | output emotion: '+output_bucket_e

        print('\n\n',transfer_input,'\n\n')
        
        t = transfer_tokenizer(transfer_input, return_tensors='pt')
        gen = transfer_model.generate(input_ids= t.input_ids, attention_mask = t.attention_mask, max_length=70, 
                                            num_beams=12,
                                            #    early_stopping=True,
                                            encoder_no_repeat_ngram_size=5,
                                            no_repeat_ngram_size=3,
                                            num_beam_groups=3,
                                            diversity_penalty=0.5,
                                            num_return_sequences=int(controls['suggestions'])
                                            )
        transfers = transfer_tokenizer.batch_decode(gen, skip_special_tokens=True)

        res = {
            'input' : {
                'text' : text,
                'probs' : {
                    'formality' : classifier_output['formality']['prob'],
                    'emo' : classifier_output['emo']['prob']
                },
            },
            "goal" : f"Formality : {output_bucket_f}; Emotion : {output_bucket_e}",
            "suggestions":[]
        }
        for transfer in transfers:
            cls_opt = classifier_trainer.predict_for_sentence(transfer, classifier_tokenizer, salience=False)
            temp = {
                'text' : transfer,
                'probs' : {
                    'formality' : cls_opt['formality']['prob'],
                    'emo' : cls_opt['emo']['prob']
                }
            }
            res['suggestions'].append(temp)
        
    elif mode=="macro-binary":
        transfer_input = 'transfer: ' + lower
        print('\n\n',transfer_input,'\n\n')
        t = transfer_tokenizer(transfer_input, return_tensors='pt')
        
        if int(controls['macro']) == 0:
            gen = transfer_model_wiki.generate(input_ids= t.input_ids, attention_mask = t.attention_mask, max_length=70, 
                                            num_beams=12,
                                            #    early_stopping=True,
                                            encoder_no_repeat_ngram_size=5,
                                            no_repeat_ngram_size=3,
                                            num_beam_groups=3,
                                            diversity_penalty=0.5,
                                            num_return_sequences=int(controls['suggestions'])
                                            )
        elif int(controls['macro']) == 1:
            gen = transfer_model_shake.generate(input_ids= t.input_ids, attention_mask = t.attention_mask, max_length=70, 
                                            num_beams=12,
                                            #    early_stopping=True,
                                            encoder_no_repeat_ngram_size=5,
                                            no_repeat_ngram_size=3,
                                            num_beam_groups=3,
                                            diversity_penalty=0.5,
                                            num_return_sequences=int(controls['suggestions'])
                                            )
        elif int(controls['macro']) == 2:
            gen = transfer_model_abs.generate(input_ids= t.input_ids, attention_mask = t.attention_mask, max_length=70, 
                                            num_beams=12,
                                            #    early_stopping=True,
                                            encoder_no_repeat_ngram_size=5,
                                            no_repeat_ngram_size=3,
                                            num_beam_groups=3,
                                            diversity_penalty=0.5,
                                            num_return_sequences=int(controls['suggestions'])
                                            )
         
        transfers = transfer_tokenizer.batch_decode(gen, skip_special_tokens=True)

        res = {
            'input' : {
                'text' : text,
            },
            "goal" : ["Wikipedia", "Shakespeare", "Scientific Abstract"][int(controls['macro'])],
            "suggestions":[]
        }
        for transfer in transfers:
            temp = {
                'text' : transfer,
            }
            res['suggestions'].append(temp)
    return res, 200



if __name__ == '__main__':
    load_models('micro-formality')
    app.run(host="0.0.0.0", port=5001, debug=True)
