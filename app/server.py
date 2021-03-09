from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates'
        )
CORS(app)


@app.route("/hello")
def hello():
    res = {
        "world": 42
    }
    return res

@app.route("/")
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
