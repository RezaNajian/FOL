import sys
import datetime

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message) 
        self.log.write(message) 

    def flush(self):
        self.terminal.flush()
        self.log.flush()