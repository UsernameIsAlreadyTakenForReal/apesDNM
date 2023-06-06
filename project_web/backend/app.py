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


class Output:
    def __init__(self):
        self._x = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if self._x != value:
            self._x = value
            self.trigger_action()

    def trigger_action(self):
        print("hello?")
        gevent.spawn(socketio.emit("console", str(self._x), broadcast=True))


output = Output()


@app.route("/testing", methods=["GET", "POST"])
def http_call():
    global output
    output.x = 0

    sleep(1)
    output.x = 1

    sleep(1)
    output.x = 2

    sleep(1)
    output.x = 3

    return "ok"


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
