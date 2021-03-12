# import flask related modules
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

# basic imports
import json
import sys

#print("\n\n\n", sys.path, "\n\n")

# Pytorch imports
import torch
from torchtext.data.utils import get_tokenizer

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import our ML code
sys.path.append("../ml")
from attention.selfAttention import SelfAttention 
import attention.attention_utils
import bert.bert_utils
from attention.utils import load_pytorch_model, load_vocab_dict
from config.model_config import get_configs
#from attention.config import batch_size, max_length



app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates'
        )
CORS(app)


configs, class_labels_dict = get_configs()
model_type = configs['model_type']

if model_type == "self_attention":
	batch_size = configs['batch_size'] 
	max_length = configs['max_length']
	vocab_dict_path = configs['vocab_dict_path'] 
	model_path = configs['model_path']
	vocab_size = configs['vocab_size']
	embedding_length = configs['embedding_length']
	hidden_size = configs['hidden_size']
	output_size = configs['output_size']

	# Load vocab_dict
	vocab_dict = load_vocab_dict(vocab_dict_path)
	# load pytorch model
	weights = torch.zeros((vocab_size, embedding_length))
	model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, 
                    embedding_length, weights)
	model = load_pytorch_model(model, model_path)
	# Load tokenizer
	tokenizer = get_tokenizer('basic_english')
	
elif model_type == "bert":
	batch_size = configs['batch_size'] 
	model_path = configs['model_path']
	model = AutoModelForSequenceClassification.from_pretrained(model_path)
	tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", model_max_length=64)
	
# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


@app.route("/hello")
def hello():
    res = {
        "world": 42
    }
    return res

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/heatmap', methods = ['GET'])
def get_classify_and_attn():
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
	if model_type == "self_attention":
		# tokenize text
		tokens = tokenizer(text) 
		# Get model outputs and attention matrix
		output, atten = attention.attention_utils.sent_pred(text, model, vocab_dict, 
						        tokenizer, max_length, device, batch_size)
		# Sum over attention vectorto get single value for each token
		token_attentions = atten[0].sum(axis=0).cpu().detach().tolist()[:len(tokens)]
		# Get model's class prediction
		preds = output.argmax(axis=1)
		pred = preds[0].item()
		output = output[0]
		softmax = torch.nn.Softmax(dim=0)
		scores = softmax(torch.tensor(output))
		
		response = {}
		response['tokens'] = tokens
		response['class id'] = pred
		response['class'] = class_labels_dict[pred]
		response['scores'] = f"Class Scores: {class_labels_dict[0]} : {scores[0]*100:.2f}%, {class_labels_dict[1]} : {scores[1]*100:.2f}%"
		response['attns'] = token_attentions
		
		# Return json response and 201 HTTP code
		print(response)
		return json.dumps(response), 201
	  
	elif model_type == "bert":
		tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))[1:-1]
		pred, scores, attns = bert.bert_utils.sent_pred(text, model, 
														 tokenizer, device, batch_size)
		pred, scores, attns = bert.bert_utils.process_outputs(pred, scores, attns)	
		response = {}
		response['tokens'] = tokens
		response['class id'] = int(pred)
		response['class'] = class_labels_dict[pred]	
		response['scores'] = f"Class Scores: {class_labels_dict[0]} : {scores[0]*100:.2f}%, {class_labels_dict[1]} : {scores[1]*100:.2f}%"
		response['attns'] = attns
		print(response)
		return json.dumps(response), 201
  
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
