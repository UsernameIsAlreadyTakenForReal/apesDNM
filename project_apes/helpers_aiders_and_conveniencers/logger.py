from datetime import datetime


class Logger:
    def __init__(self):
        pass

    def info(self, sender, text_to_log):
        print(str(datetime.now()) + " -- " + str(sender) + " -- " + text_to_log)
