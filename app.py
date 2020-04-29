from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/home')
def home():
    return '<h1>You are on the home page!</h1>'

@app.route('/json')
def json():
    return jsonify({'key' : 'value'})
if __name__ == "__main__":
    app.run()