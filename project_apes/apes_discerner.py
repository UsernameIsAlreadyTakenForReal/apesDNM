# This file:
# Defines a class meant to figure out what kind of data the given dataset contains.

import sys
import apes_static_definitions


class Discerner:
    def __init__(self, Logger):
        self.Logger = Logger

        info_message = "Created object of type Discerner."
        self.Logger.info(self, info_message)
