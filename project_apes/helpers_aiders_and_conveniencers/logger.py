from datetime import datetime


class Logger:
    def __init__(self):
        pass

    def info(self, text_to_log):
        print(str(datetime.now()) + " -- " + text_to_log)
