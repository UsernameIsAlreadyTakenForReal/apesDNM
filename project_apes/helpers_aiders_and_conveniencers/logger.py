from datetime import datetime
import gevent
import time


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    _instance = None

    def __init__(self, socketio=None, skip_print=False):
        self.socketio = socketio
        self.message_in_queue = (
            "Nothing to print in Logger. This may mean something is wrong."
        )
        self.skip_print = False
        pass

    def info(self, sender, text_to_log):
        message = str(datetime.now()) + " -- " + str(sender) + " -- " + text_to_log
        print(message)

        if self.socketio is not None:
            gevent.spawn(self.socketio.emit("console", str(message)))

        time.sleep(0.1)

    def print_info(self):
        print(self.message_in_queue)
