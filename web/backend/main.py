from flask import Flask
from flask_cors import CORS, cross_origin
from flask_restful import Api, Resource

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'
api = Api(app)

class Testing(Resource):
    def get(self):
        return " -- string returned from backend -- "

api.add_resource(Testing, "/testing")

if __name__ == "__main__":
    app.run(debug=True)