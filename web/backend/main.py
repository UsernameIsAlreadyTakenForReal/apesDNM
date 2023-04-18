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

@app.route('/double', methods=['POST'])
def get_double():
    json1 = json.loads(json.dumps(request.get_json()))
    print(json1["value1"] + "; " +  json1["value2"])
    return { "message": "sum is " + str(int(json1["value1"]) + int(json1["value2"])) }

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    print("hello from upload_files()")
    default_value = 'alabala'
    file = request.form.get("file") or default_value
    file_the_good_one = request.files.get("file")
    # https://stackoverflow.com/questions/51765498/flask-file-upload-unable-to-get-form-data
    return file_the_good_one

if __name__ == "__main__":
    app.run(debug=True, port=5000)