# import flask related modules
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import requests

# basic imports
import json
import sys

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates'
        )
CORS(app)


@app.route("/hello")
def hello():
    res = {
        "world": 42,
		"server": "app"
    }
    return res

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/stats', methods = ['GET'])
def get_stats():
	'''
	Inputs:
	Input is assumed to be json of the form 
  	{text: "some text"}. 
  	
  	Returns:
	Following statistics for the text :
	- Attention
		- Class, class probs
		- Attention distribution
  	'''
	# Get text input from request
	text = request.args.get('text', type = str)

	# Get attention heatmap
	# response_attn = requests.get('http://0.0.0.0:5001/heatmap', params={'text': text}).json()
	response_joint = requests.get('http://0.0.0.0:5001/joint_classification', params={'text': text}).json()
	response_joint['input'] = text

	# return {'attn' : response_attn, 'joint' : response_joint}, 201
	return {'joint' : response_joint}, 201
  
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
