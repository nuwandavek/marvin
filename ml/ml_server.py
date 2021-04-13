# import flask related modules
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

# basic imports
import json
import sys

# Pytorch imports
import torch
from torchtext.data.utils import get_tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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


# # Load spacy for tokenizing
# import spacy
# # Load English tokenizer, tagger, parser and NER
# nlp = spacy.load("en_core_web_sm")

# json.encoder.FLOAT_REPR = lambda o: format(o, '.2f')

app = Flask(__name__)
CORS(app)


# configs, class_labels_dict = get_configs()
# model_type = configs['model_type']

# if model_type == "self_attention":
#     batch_size = configs['batch_size'] 
#     max_length = configs['max_length']
#     vocab_dict_path = configs['vocab_dict_path'] 
#     model_path = configs['model_path']
#     vocab_size = configs['vocab_size']
#     embedding_length = configs['embedding_length']
#     hidden_size = configs['hidden_size']
#     output_size = configs['output_size']

#     # Load vocab_dict
#     vocab_dict = load_vocab_dict(vocab_dict_path)
#     # load pytorch model
#     weights = torch.zeros((vocab_size, embedding_length))
#     model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, 
#                     embedding_length, weights)
#     model = load_pytorch_model(model, model_path)
#     # Load tokenizer
#     tokenizer = get_tokenizer('basic_english')
    
# elif model_type == "bert":
#     batch_size = configs['batch_size'] 
#     model_path = configs['model_path']
#     model = AutoModelForSequenceClassification.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", model_max_length=64)
    
# # Use GPU if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

model_args = ModelArguments(
    model_name_or_path="distilbert-base-uncased",
    model_nick="distilbert_uncased",
    cache_dir="./models/cache"
)

data_args = DataTrainingArguments(
    max_seq_len=64,
    task="formality+jokes"
)

training_args = TrainingArguments(
    output_dir = "./models/distilbert_uncased/formality+jokes/joint",
    train_jointly= True
)
idx_to_classes = {
    'formality': {'0': 'informal', '1': 'formal'},
    'jokes': {'0': 'nojoke', '1': 'joke'}
}
label_dims = {'formality': 1, 'jokes': 1}

model_joint = JointSeqClassifier.from_pretrained(
    training_args.output_dir,
    tasks=data_args.task.split('+'),
    model_args=model_args,
    task_if_single=None, 
    joint = training_args.train_jointly,
    label_dims=label_dims
)
trainer_joint = JointTrainer(
    [training_args,model_args, data_args], 
    model_joint, idx_to_classes = idx_to_classes
)
tokenizer_joint = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path, 
    cache_dir=model_args.cache_dir,
    model_max_length = data_args.max_seq_len
)

@app.route("/hello")
def hello():
    res = {
        "world": 42,
        "app": "ml"
    }
    return res

# @app.route('/heatmap', methods = ['GET'])
# def get_classify_and_attn():
#     '''
#     Inputs:
#     Input is assumed to be json of the form 
#       {text: "some text"}.
  
#       Results:
#       Run ML classification model on text. 
      
#       Returns:
#     classification and attention weights
#       '''
#     # Get text input from request
#     text = request.args.get('text', type = str)
#     text = text.strip()
#     if model_type == "self_attention":
#         # tokenize text
#         tokens = tokenizer(text) 
#         # Get model outputs and attention matrix
#         output, atten = attention.attention_utils.sent_pred(text, model, vocab_dict, 
#                                 tokenizer, max_length, device, batch_size)
#         # Sum over attention vector to get single value for each token
#         token_attentions = atten[0].sum(axis=0).cpu().detach().tolist()[:len(tokens)]
#         # Get model's class prediction
#         preds = output.argmax(axis=1)
#         pred = preds[0].item()
#         output = output[0]
#         softmax = torch.nn.Softmax(dim=0)
#         scores = softmax(torch.tensor(output))
        
#         response = {}
#         response['tokens'] = tokens
#         response['class id'] = pred
#         response['class'] = class_labels_dict[pred]
#         response['scores'] = f"Class Scores: {class_labels_dict[0]} : {scores[0]*100:.2f}%, {class_labels_dict[1]} : {scores[1]*100:.2f}%"
#         response['attns'] = token_attentions
        
#         # Return json response and 201 HTTP code
#         print(response)
#         return json.dumps(response), 201
      
#     elif model_type == "bert":
#         doc = nlp(text)
#         tokens = []
#         sentence_seen = 0

#         for token in [t.text for t in doc]:    
#             occ = text[sentence_seen:].find(token)
#             # Handle the case where the first character in the tokenm is an apostrophe, since BERT tokenizers 
#             # from HuggingFace create a separate apostrophe token but SpaCy does not
#             if token[0] != '\'':
#                 start = occ + sentence_seen
#                 end = start + len(token)
#                 sentence_seen = sentence_seen + len(token) + occ
#                 tokens.append({'text' : token, 'start' : start, 'end' : end})
#             else:
#                 start_apos = occ + sentence_seen
#                 start_rest = start_apos + 1
#                 end_apos = start_apos + 1
#                 end_rest = start_rest + (len(token) - 1)
#                 sentence_seen = sentence_seen + len(token) + occ
#                 tokens.append({'text' : token[0], 'start' : start_apos, 'end' : end_apos})
#                 tokens.append({'text' : token[1:], 'start' : start_rest, 'end' : end_rest})

#         bert_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))[1:-1]
#         bert_token_mask = [i for i,b in enumerate(bert_tokens) if '##' not in b]
        
#         print(f'bert tokens {bert_tokens}')
        
#         pred, scores, attns = bert.bert_utils.sent_pred(text, model, tokenizer, device, batch_size)
#         pred, scores, tokens = bert.bert_utils.process_outputs(pred, scores, attns, bert_token_mask, tokens)    
#         response = {}
#         response['text'] = text
#         response['tokens'] = tokens
#         response['class id'] = int(pred)
#         response['class'] = class_labels_dict[pred]    
#         response['scores'] = f"Class Scores: {class_labels_dict[0]} : {scores[0]*100:.2f}%, {class_labels_dict[1]} : {scores[1]*100:.2f}%"
#         print(f"Heatmap RES\n{response}")
#         return json.dumps(response), 201
  

@app.route('/joint_classification', methods = ['GET'])
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

    joint_tokens = tokenizer_joint.convert_ids_to_tokens(tokenizer_joint.encode(lower))[1:-1]
    for token in joint_tokens:
        occ = lower[sentence_seen:].find(token)
        start = occ + sentence_seen
        end = start + len(token)
        sentence_seen = sentence_seen + len(token) + occ
        tokens.append({'text' : text[start:end], 'start' : start, 'end' : end})
    
    
    res = trainer_joint.predict_for_sentence(lower, tokenizer_joint, salience=True)
    res['tokens'] = tokens
    # print(f"JointClassify RES\n{res}")
    return res, 200



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
