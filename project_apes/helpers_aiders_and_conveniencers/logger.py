from datetime import datetime
import gevent


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    _instance = None

    # def __new__(cls, *args, **kwargs):
    #     if not cls._instance:
    #         cls._instance = super().__new__(cls, *args, **kwargs)
    #     return cls._instance

    def __init__(self, socketio):
        self.socketio = socketio
        self.message_in_queue = (
            "Nothing to print in Logger. This may mean something is wrong."
        )
        pass

    def info(self, sender, text_to_log):
        message = str(datetime.now()) + " -- " + str(sender) + " -- " + text_to_log
        gevent.spawn(self.socketio.emit("console", str(message), broadcast=True))
        print(message)

    def print_info(self):
        print(self.message_in_queue)
