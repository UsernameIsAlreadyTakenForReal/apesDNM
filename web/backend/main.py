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

from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'

@app.route('/morbin')
def its_morbin_time():
    return {"value": "its morbin time"}

@app.route('/double', methods=['GET', 'POST'])
def get_double():
    value = request.json.get('value')
    if value == "": value = 0
    new_val = {"value": 2*int(value)}
    return new_val

@app.route('/upload', methods = ['POST'])
def upload_file():
    file = request.files['file']
    print(file)
    return "done"

if __name__ == "__main__":
    app.run(debug=True, port=5000)