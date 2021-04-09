#! /usr/bin/python3

from flask import Flask, render_template, url_for, flash, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates'
        )

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True) #since Id is primary key, it will be assigned automatically
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique = True, nullable=False)
    password = db.Column(db.String(60), nullable=False)  #Hashing algorithm will make the password 60 char long
    documents = db.relationship('Doc', backref='author', lazy=True) #This will help us fetch all the docs attributes generated by this user
    date_created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"


class Doc(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    time_stamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    styles = db.relationship('Style', backref='document', lazy=True)
    saliencies = db.relationship('Saliency', backref='document', lazy=True)

    def __repr__(self):
        return f"Doc('{self.title}', '{self.time_stamp}')"

class Style(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    docid = db.Column(db.Integer, db.ForeignKey('doc.id'), nullable=False)
    input = db.Column(db.Text, nullable=False)
    output = db.Column(db.Text, nullable=False)
    intention = db.Column(db.String(100), nullable=False)
    time_stamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


    def __repr__(self):
        return f"Post('{self.input}', '{self.output}', '{self.intention}')"

class Saliency(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    docid = db.Column(db.Integer, db.ForeignKey('doc.id'), nullable=False)
    input = db.Column(db.Text, nullable=False)
    output = db.Column(db.Text, nullable=False)
    time_stamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


    def __repr__(self):
        return f"Post('{self.input}', '{self.output}')"








