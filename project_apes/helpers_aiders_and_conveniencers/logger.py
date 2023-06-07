from datetime import datetime


class Logger:
    def __init__(self, skip_print=False):
        self.message_in_queue = (
            "Nothing to print in Logger. This may mean something is wrong."
        )
        self.skip_print = skip_print
        pass

    def info(self, sender, text_to_log):
        if self.skip_print == False:
            print(str(datetime.now()) + " -- " + str(sender) + " -- " + text_to_log)

    def print_info(self):
        print(self.message_in_queue)
