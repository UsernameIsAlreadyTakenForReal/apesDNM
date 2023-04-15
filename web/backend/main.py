# from flask import Flask, request
# from flask_cors import CORS, cross_origin
# from flask_restful import Api, Resource

# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADER'] = 'Content-Type'
# api = Api(app)

# class Testing(Resource):
#     def get(self):
#         return "this is the response of a get method call"
#     def post(self):
#         value = request.json.get('value')
#         if value == "":
#             value = 0
#         new_val = {"value": int(value)*2}
#         print(new_val)
#         return new_val
    
# class Morbius(Resource):
#     def get(self):
#         return "watching him shout its morbin time! and starting to morb all the enemies was such a morbelous scene"

# api.add_resource(Testing, "/testing")
# api.add_resource(Morbius, "/morbin")

# if __name__ == "__main__":
#     app.run(debug=True)

#############################################################################
# this this kinda works but also acts a bit weird so hmmmm???

import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

@app.route('/morbin')
def its_morbin_time():
    return jsonify("this is a string")

@app.route('/double', methods=['GET', 'POST'])
def get_double():
    value = request.json.get('value')
    if value == "": value = 0
    new_val = {"msg": 2*int(value)}
    return new_val

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    # if request.method == 'POST':
    #     f = request.files['file']
    #     f.save(secure_filename(f.filename))
    return "file uploaded successfully"

@app.route('/uploaddd', methods = ['GET', 'POST'])
def uploaddd_file():
    record = json.loads(request.data)
    with open('/tmp/data.txt', 'r') as f:
        data = f.read()
    if not data:
        records = [record]
    else:
        records = json.loads(data)
        records.append(record)
    with open('/tmp/data.txt', 'w') as f:
        f.write(json.dumps(records, indent=2))
    return jsonify(record)

if __name__ == "__main__":
    app.run(debug=True, port=5000)