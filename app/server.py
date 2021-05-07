# import flask related modules
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import requests
import pymysql
# basic imports
import json
import sys

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates'
        )
CORS(app)

con = pymysql.connect(host='0.0.0.0', user='root', password='root', database='marvin')

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

@app.route('/swap_models', methods=['POST'])
def swap_models():
	mode = request.form.get('mode', type = str)
	print(mode)
	response_joint = requests.post('http://0.0.0.0:5001/swap_models', params={'mode': mode})
	if response_joint.status_code==200:
		return {'message' : 'Models Successfully Swapped!'}, 200
	else:
		return {'message' : 'Models Swap Failure! :('}, 500

@app.route('/analyze', methods = ['GET'])
def get_stats():
	'''
	Inputs:
	Input is assumed to be json of the form 
  	{text: "some text"}. 
  	
  	Returns:
	Following statistics for the text :
	- Class, class probs
	- Salience over tokens for each class
  	'''
	# Get text input from request
	text = request.args.get('text', type = str)
	mode = request.args.get('mode', type = str)

	# Get attention heatmap
	response_joint = requests.get('http://0.0.0.0:5001/classification', params={'text': text, 'mode': mode}).json()
	response_joint['input'] = text

	return {'results' : response_joint}, 200

@app.route('/transfer', methods = ['GET'])
def get_transfer_suggestions():
	# Get text input from request
	text = request.args.get('text', type = str).strip()
	controls = request.args.get('controls', type = str)
	mode = request.args.get('mode', type = str)
	
	
	response_transfer = requests.get('http://0.0.0.0:5001/transfer', params={'text': text, 'mode': mode, 'controls':controls}).json()
	return response_transfer, 200

@app.route('/transfer_action', methods = ['POST'])
def transfer_action():
	original = request.form.get('original', type = str).strip()
	original_val = '|'.join([k+":"+str(v) for k,v in json.loads(request.form.get('original_val', type = str).strip()).items()])
	accepted = request.form.get('accepted', type = str).strip()
	accepted_val = '|'.join([k+":"+str(v) for k,v in json.loads(request.form.get('accepted_val', type = str).strip()).items()])
	mode = request.form.get('mode', type = str).strip()
	goal = request.form.get('goal', type = str).strip()
	
	print('\n\n')
	print(mode, goal)
	print(original, original_val)
	print(accepted, accepted_val)
	print('\n\n')
	with con.cursor() as cur:
		cur.execute(f'INSERT INTO accepted_transfers (mode, goal, original, original_val, accepted, accepted_val) VALUES ("{mode}", "{goal}", "{original}", "{original_val}","{accepted}", "{accepted_val}")')
	con.commit()
	with con.cursor() as cur:
		cur.execute(f'SELECT LAST_INSERT_ID()')
	row = cur.fetchone()
	
	return {'id' : row[0]} , 201

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
