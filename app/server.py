# import flask related modules
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

# basic imports
import json
import sys

# Pytorch imports
import torch
from torchtext.data.utils import get_tokenizer

# Import our ML code
sys.path.append("../ml")
from attention.selfAttention import SelfAttention
from attention.attention_utils import sent_embedd, sent_pred
from attention.utils import load_pytorch_model, load_vocab_dict
from attention.config import batch_size, max_length

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates'
        )
CORS(app)



# Load vocab_dict
vocab_dict = load_vocab_dict("../ml/models/attention/vocab_dict.pickle")


# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load pytorch model
model = load_pytorch_model("../ml/models/attention/attn_test.pth")
model.to(device)

# Load tokenizer
#tokenizer = lambda x : x.split()
tokenizer = get_tokenizer('basic_english')


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
  # tokenize text
  tokens = tokenizer(text) 
  # Get model outputs and attention matrix
  output, attention = sent_pred(text, model, vocab_dict, 
                                tokenizer, max_length, device, batch_size)
  # Sum over attention vectorto get single value for each token
  token_attentions = attention[0].sum(axis=0).cpu().detach().tolist()[:len(tokens)]
  # Get model's class prediction
  preds = output.argmax(axis=1)
  pred = preds[0].item()
  
  # Return json response and 201 HTTP code
  return json.dumps({'class' : pred, 'tokens' : tokens, 'scores' : token_attentions}), 201
  
  
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
