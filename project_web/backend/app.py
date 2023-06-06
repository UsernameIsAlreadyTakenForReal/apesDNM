from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import gevent

from gevent import monkey

monkey.patch_all()

from time import sleep

app = Flask(__name__)
app.config["SECRET_KEY"] = "apesDNM"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")
CORS(app)

output = 0


@app.route("/testing", methods=["GET", "POST"])
def http_call():
    global output
    output = 0
    gevent.spawn(socketio.emit("console", str(output), broadcast=True))

    sleep(1)
    output = 1
    gevent.spawn(socketio.emit("console", str(output), broadcast=True))

    sleep(1)
    output = 2
    gevent.spawn(socketio.emit("console", str(output), broadcast=True))

    sleep(1)
    output = 3
    gevent.spawn(socketio.emit("console", str(output), broadcast=True))

    return "ok"


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
