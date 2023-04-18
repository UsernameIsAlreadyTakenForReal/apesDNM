import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'

@app.route('/morbin')
def its_morbin_time():
    return { "message": "it's morbin time" }

@app.route('/double', methods=['GET', 'POST'])
def get_double():
    json1 = json.loads(json.dumps(request.get_json()))
    print(json1["value1"] + "; " +  json1["value2"])
    return { "message": "sum is " + str(int(json1["value1"]) + int(json1["value2"])) }

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    print("we got here yes")
    default_value = '0'
    file = request.form.get("file", default_value)
    return file

if __name__ == "__main__":
    app.run(debug=True, port=5000)