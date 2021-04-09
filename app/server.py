# import flask related modules
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import requests
from tables import db, app
from tables import User, Doc, Style, Saliency
from flask_bcrypt import Bcrypt
import sys,json


CORS(app)
bcrypt = Bcrypt(app)

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
	response_attn = requests.get('http://0.0.0.0:5001/heatmap', params={'text': text}).json()
	response_joint = requests.get('http://0.0.0.0:5001/joint_classification', params={'text': text}).json()

	return {'attn' : response_attn, 'joint' : response_joint}, 201

#Register a new user
@app.route("/register", methods=['POST'])
def register():
    data = request.get_json() #SAMPLE: {"username": "Yoyoy", "password":"ghx", "email":"axyz@abc.com"}

    username = data['username']
    #Check if username already exists
    user_list = User.query.filter_by(username=str(username)).all()
    if(len(user_list)>0):
        response = {"error": str(username)+" already exists."}
        return json.dumps(response), 404
    
    email = data['email']
    #Check if email already exists
    email_list = User.query.filter_by(email=str(email)).all()
    if(len(email_list)>0):
        response = {"error": str(email)+" already exists."}
        return json.dumps(response), 404

    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    user = User(username=username, email=email, password=hashed_password)
    db.session.add(user)
    db.session.commit()
    response = {"Success": "User registered successfully."}
    return json.dumps(response), 200

#Login for the user
@app.route("/login", methods=['POST'])
def login():
    data = request.get_json() #SAMPLE:{"username": "Yoyoy", "password":"ghx"}
    username = data['username']

    user = User.query.filter_by(username=username).first()
    if(user!=None and bcrypt.check_password_hash(user.password, data['password'])):
        response = {"Success": "User authenticated successfully"}
        return json.dumps(response), 200
    else:
        response = {"error": 'Login Unsuccessful. Please check username and password'}
        return json.dumps(response), 404

#Create a new doc
@app.route("/doc/new", methods=['POST'])
def new_doc():
    data = request.get_json() #SAMPLE:{"title":"sample", "content":"sample", "username":"yoyo"}
    title = data['title']
    content=data['content']
    username=data['username']
    user = User.query.filter_by(username=str(username)).first()

    #Check if this title already exists for this user
    for doc in user.documents:
        if(doc.title == title):
            response = {"error":"This title already exists"}
            return json.dumps(response), 404

    new_doc = Doc(title=title, content=content, user_id = user.id)
    db.session.add(new_doc)
    db.session.commit()
    response = {"Success": "Document created"}
    return json.dumps(response), 200

#Get all the docs for a user
@app.route("/doc/<username>", methods=['GET'])
def getDoc(username):
    user = User.query.filter_by(username=str(username)).first()
    if(len(user.documents)>0):
        response = {'docs':[]}
        for doc in user.documents:
            response['docs'].append({"docid":doc.id, "title":doc.title})
        return json.dumps(response), 200
    else:
        response = {"error": "No documents found for this user"}
        return json.dumps(response), 404 

#Get doc content for a user
@app.route("/doc_content/<docid>", methods=['GET'])
def getDocContent(docid):
    req_doc = Doc.query.get_or_404(int(docid)) #If not found, it returns 404 error
    response = {"title": req_doc.title, "content":req_doc.content}
    return json.dumps(response), 200 

#Update doc content for a user
@app.route("/doc_content/<docid>", methods=['POST'])
def updateDoc(docid):
    req_doc = Doc.query.get_or_404(int(docid))
    data = request.get_json() #SAMPLE:{"title":"sample", "content":"sample"}
    title=data['title']
    content=data['content']
    req_doc.title = title
    req_doc.content = content
    db.session.commit()
    response = {"Success": "Document updated successfully"}
    return json.dumps(response), 200

#Delete doc 
@app.route("/doc/<docid>", methods=['DELETE'])
def deleteDoc(docid):
    req_doc = Doc.query.get_or_404(int(docid))
    db.session.delete(req_doc)
    db.session.commit()
    response = {"Success":"Document deleted successfully"}
    return json.dumps(response), 200

  
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
